#!/usr/bin/env python3
"""
Unit Tests for /api/upcoming_races_csv Endpoint
==============================================

Tests the API endpoint that lists upcoming races from CSV files,
including pagination, search, and data structure validation.

Created as part of Step 6: Unit & integration tests
"""

import hashlib
import os
import shutil

import pytest

# Import test fixtures


class TestUpcomingRacesCSVEndpoint:
    """Unit tests for /api/upcoming_races_csv endpoint"""

    @pytest.fixture
    def temp_upcoming_dir(self, test_app):
        """Create temporary upcoming_races directory with test CSV files"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create three dummy CSV files with different header structures
        test_csv_files = [
            {
                "filename": "Race_1_WPK_2025-02-01.csv",
                "content": """Race Name,Venue,Race Date,Distance,Grade,Race Number,Dog Name,Box,Weight,Trainer
Test Race One,WPK,2025-02-01,500m,Grade 5,1,Test Dog Alpha,1,30.2,Trainer Smith
Test Race One,WPK,2025-02-01,500m,Grade 5,1,Test Dog Beta,2,29.8,Trainer Jones
Test Race One,WPK,2025-02-01,500m,Grade 5,1,Test Dog Gamma,3,30.5,Trainer Brown""",
            },
            {
                "filename": "Race_2_MEA_2025-02-02.csv",
                "content": """Venue,Race_Date,Race_Name,Distance,Grade,Race_Number
MEA,2025-02-02,Test Race Two,520m,Grade 4,2
MEA,2025-02-02,Test Race Two,520m,Grade 4,2
MEA,2025-02-02,Test Race Two,520m,Grade 4,2""",
            },
            {
                "filename": "GOSF_Race_3_2025-02-03.csv",
                "content": """Dog Name,Box Number,Weight,Trainer Name
Test Dog Delta,1,31.0,Trainer Wilson
Test Dog Echo,2,29.5,Trainer Davis
Test Dog Foxtrot,3,30.8,Trainer Taylor
Test Dog Golf,4,30.1,Trainer Anderson""",
            },
        ]

        # Write test CSV files
        for csv_data in test_csv_files:
            file_path = os.path.join(upcoming_dir, csv_data["filename"])
            with open(file_path, "w", newline="") as f:
                f.write(csv_data["content"])

        yield upcoming_dir, test_csv_files

        # Cleanup is handled by test_app fixture

    def test_upcoming_races_csv_basic_functionality(self, client, temp_upcoming_dir):
        """Test basic functionality of /api/upcoming_races_csv endpoint"""
        upcoming_dir, test_files = temp_upcoming_dir

        response = client.get("/api/upcoming_races_csv")
        assert response.status_code == 200

        data = response.get_json()
        assert data["success"] is True
        assert "races" in data
        assert "pagination" in data
        assert isinstance(data["races"], list)

        # Should return all 3 test CSV files
        assert len(data["races"]) == 3

    def test_upcoming_races_csv_correct_count(self, client, temp_upcoming_dir):
        """Test that API returns correct count of CSV files"""
        upcoming_dir, test_files = temp_upcoming_dir

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        # Verify pagination info reflects correct total count
        assert data["pagination"]["total_count"] == 3
        assert (
            data["pagination"]["total_pages"] == 1
        )  # All should fit on one page with default per_page=10
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["per_page"] == 10
        assert data["pagination"]["has_next"] is False
        assert data["pagination"]["has_prev"] is False

    def test_upcoming_races_csv_structure_validation(self, client, temp_upcoming_dir):
        """Test that API returns races with correct data structure"""
        upcoming_dir, test_files = temp_upcoming_dir

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        races = data["races"]
        assert len(races) > 0

        # Check first race structure
        race = races[0]
        required_fields = [
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "race_name",
            "grade",
            "distance",
            "field_size",
            "winner_name",
            "winner_odds",
            "winner_margin",
            "url",
            "extraction_timestamp",
            "track_condition",
            "runners",
            "filename",
        ]

        for field in required_fields:
            assert field in race, f"Required field '{field}' missing from race data"

        # Verify data types and default values for upcoming races
        assert isinstance(race["race_id"], str)
        assert isinstance(race["venue"], str)
        assert isinstance(race["race_number"], int)
        assert isinstance(race["race_date"], str)
        assert isinstance(race["race_name"], str)
        assert isinstance(race["grade"], str)
        assert isinstance(race["distance"], str)
        assert isinstance(race["field_size"], int)
        assert race["winner_name"] == "Unknown"  # Upcoming race, no winner
        assert race["winner_odds"] == "N/A"
        assert race["winner_margin"] == "N/A"
        assert race["url"] == ""  # No URL for upcoming races
        assert isinstance(race["extraction_timestamp"], str)
        assert race["track_condition"] == "Unknown"
        assert isinstance(race["runners"], list)
        assert isinstance(race["filename"], str)
        assert race["filename"].endswith(".csv")

    def test_races_response_is_list_array(self, client, temp_upcoming_dir):
        """Unit test: Assert that response.json()['races'] is a list (Task requirement)"""
        upcoming_dir, test_files = temp_upcoming_dir

        response = client.get("/api/upcoming_races")
        assert response.status_code == 200

        # Task requirement: assert isinstance(response.json()['races'], list)
        assert isinstance(response.get_json()["races"], list)

    def test_upcoming_races_csv_header_parsing(self, client, temp_upcoming_dir):
        """Test that different CSV header formats are parsed correctly"""
        upcoming_dir, test_files = temp_upcoming_dir

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        races = data["races"]
        race_by_filename = {race["filename"]: race for race in races}

        # Test first CSV with full headers
        race1 = race_by_filename["Race_1_WPK_2025-02-01.csv"]
        assert race1["race_name"] == "Test Race One"
        assert race1["venue"] == "WPK"
        assert race1["race_date"] == "2025-02-01"
        assert race1["distance"] == "500m"
        assert race1["grade"] == "Grade 5"
        assert race1["race_number"] == 1

        # Test second CSV with underscore headers
        race2 = race_by_filename["Race_2_MEA_2025-02-02.csv"]
        assert race2["race_name"] == "Test Race Two"
        assert race2["venue"] == "MEA"
        assert race2["race_date"] == "2025-02-02"
        assert race2["distance"] == "520m"
        assert race2["grade"] == "Grade 4"
        assert race2["race_number"] == 2

        # Test third CSV with minimal headers (should use filename extraction)
        race3 = race_by_filename["GOSF_Race_3_2025-02-03.csv"]
        assert race3["venue"] == "GOSF"  # Extracted from filename
        assert race3["race_date"] == "2025-02-03"  # Extracted from filename
        assert race3["race_number"] == 3  # Extracted from filename

    def test_upcoming_races_csv_pagination(self, client, temp_upcoming_dir):
        """Test pagination functionality"""
        upcoming_dir, test_files = temp_upcoming_dir

        # Test with per_page=2 to force pagination
        response = client.get("/api/upcoming_races_csv?page=1&per_page=2")
        assert response.status_code == 200

        data = response.get_json()
        assert data["success"] is True
        assert len(data["races"]) == 2  # Should return 2 races on first page
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["per_page"] == 2
        assert data["pagination"]["total_count"] == 3
        assert data["pagination"]["total_pages"] == 2
        assert data["pagination"]["has_next"] is True
        assert data["pagination"]["has_prev"] is False

        # Test second page
        response = client.get("/api/upcoming_races_csv?page=2&per_page=2")
        data = response.get_json()
        assert len(data["races"]) == 1  # Should return 1 race on second page
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["has_next"] is False
        assert data["pagination"]["has_prev"] is True

    def test_upcoming_races_csv_search_functionality(self, client, temp_upcoming_dir):
        """Test search functionality"""
        upcoming_dir, test_files = temp_upcoming_dir

        # Search by venue
        response = client.get("/api/upcoming_races_csv?search=WPK")
        data = response.get_json()
        assert data["success"] is True
        assert len(data["races"]) == 1
        assert data["races"][0]["venue"] == "WPK"

        # Search by race name
        response = client.get("/api/upcoming_races_csv?search=Test Race Two")
        data = response.get_json()
        assert len(data["races"]) == 1
        assert data["races"][0]["race_name"] == "Test Race Two"

        # Search by grade
        response = client.get("/api/upcoming_races_csv?search=Grade 5")
        data = response.get_json()
        assert len(data["races"]) == 1
        assert data["races"][0]["grade"] == "Grade 5"

        # Search by filename
        response = client.get("/api/upcoming_races_csv?search=GOSF")
        data = response.get_json()
        assert len(data["races"]) == 1
        assert "GOSF" in data["races"][0]["filename"]

        # Search with no results
        response = client.get("/api/upcoming_races_csv?search=NONEXISTENT")
        data = response.get_json()
        assert len(data["races"]) == 0
        assert data["pagination"]["total_count"] == 0

    def test_upcoming_races_csv_sorting(self, client, temp_upcoming_dir):
        """Test sorting functionality"""
        upcoming_dir, test_files = temp_upcoming_dir

        # Test sorting by race_date descending (default)
        response = client.get("/api/upcoming_races_csv?sort_by=race_date&order=desc")
        data = response.get_json()
        races = data["races"]

        # Should be sorted by date, newest first
        assert races[0]["race_date"] == "2025-02-03"
        assert races[1]["race_date"] == "2025-02-02"
        assert races[2]["race_date"] == "2025-02-01"

        # Test sorting by venue ascending
        response = client.get("/api/upcoming_races_csv?sort_by=venue&order=asc")
        data = response.get_json()
        races = data["races"]

        # Should be sorted alphabetically by venue
        venues = [race["venue"] for race in races]
        assert venues == sorted(venues)

    def test_upcoming_races_csv_edge_cases(self, client, test_app):
        """Test edge cases and error conditions"""

        # Test with no upcoming_races directory
        if os.path.exists(test_app.config["UPCOMING_DIR"]):
            shutil.rmtree(test_app.config["UPCOMING_DIR"])

        response = client.get("/api/upcoming_races_csv")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert len(data["races"]) == 0
        assert data["pagination"]["total_count"] == 0

        # Recreate directory for other tests
        os.makedirs(test_app.config["UPCOMING_DIR"], exist_ok=True)

    def test_upcoming_races_csv_invalid_parameters(self, client, temp_upcoming_dir):
        """Test invalid parameter handling"""
        upcoming_dir, test_files = temp_upcoming_dir

        # Test invalid page parameter
        response = client.get("/api/upcoming_races_csv?page=abc")
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "Invalid page or per_page parameter" in data["message"]

        # Test invalid per_page parameter
        response = client.get("/api/upcoming_races_csv?per_page=xyz")
        assert response.status_code == 400

        # Test page < 1
        response = client.get("/api/upcoming_races_csv?page=0")
        assert response.status_code == 400
        data = response.get_json()
        assert "Page number must be greater than 0" in data["message"]

        # Test per_page < 1
        response = client.get("/api/upcoming_races_csv?per_page=0")
        assert response.status_code == 400
        data = response.get_json()
        assert "Per page value must be greater than 0" in data["message"]

        # Test per_page limit (should be capped at 50)
        response = client.get("/api/upcoming_races_csv?per_page=999")
        assert response.status_code == 200
        data = response.get_json()
        assert data["pagination"]["per_page"] == 50  # Should be capped

    def test_upcoming_races_csv_race_id_generation(self, client, temp_upcoming_dir):
        """Test that race IDs are generated consistently"""
        upcoming_dir, test_files = temp_upcoming_dir

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        races = data["races"]
        race_ids = [race["race_id"] for race in races]

        # All race IDs should be unique
        assert len(race_ids) == len(set(race_ids))

        # Race IDs should be consistent for same filename (MD5 hash)
        for race in races:
            expected_race_id = hashlib.md5(race["filename"].encode()).hexdigest()[:12]
            assert race["race_id"] == expected_race_id

    def test_upcoming_races_csv_nan_value_handling(self, client, test_app):
        """Test handling of NaN and None values in CSV data"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create CSV with NaN values
        nan_csv_content = """Race Name,Venue,Race Date,Distance,Grade
,NaN,2025-02-04,null,nan"""

        nan_file_path = os.path.join(upcoming_dir, "nan_test.csv")
        with open(nan_file_path, "w") as f:
            f.write(nan_csv_content)

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        # Find the race with NaN values
        nan_race = None
        for race in data["races"]:
            if race["filename"] == "nan_test.csv":
                nan_race = race
                break

        assert nan_race is not None
        assert nan_race["race_name"] == "nan_test"  # Should fallback to filename
        assert nan_race["venue"] == "Unknown"  # Should clean NaN to Unknown
        assert nan_race["grade"] == "Unknown"  # Should clean NaN to Unknown
        assert nan_race["distance"] == "Unknown"  # Should clean null to Unknown

        # Cleanup
        os.remove(nan_file_path)

    def test_upcoming_races_csv_empty_csv_file(self, client, test_app):
        """Test handling of empty CSV files"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create empty CSV file
        empty_file_path = os.path.join(upcoming_dir, "empty_test.csv")
        with open(empty_file_path, "w") as f:
            f.write("")  # Empty file

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        # Should handle empty file gracefully without crashing
        assert response.status_code == 200
        assert data["success"] is True

        # Cleanup
        os.remove(empty_file_path)


class TestUpcomingRacesCSVIntegration:
    """Integration tests for CSV processing pipeline"""

    def test_csv_metadata_extraction_comprehensive(self, client, test_app):
        """Test comprehensive CSV metadata extraction"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create comprehensive test CSV
        comprehensive_csv = """Race Name,Venue,Race Date,Distance,Grade,Race Number,Track Condition
Melbourne Cup Trial,MEA,2025-03-15,520m,Group 1,8,Good 4"""

        file_path = os.path.join(upcoming_dir, "MEA_Race_8_2025-03-15_Group1.csv")
        with open(file_path, "w") as f:
            f.write(comprehensive_csv)

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        # Find our test race
        test_race = None
        for race in data["races"]:
            if "MEA_Race_8" in race["filename"]:
                test_race = race
                break

        assert test_race is not None
        assert test_race["race_name"] == "Melbourne Cup Trial"
        assert test_race["venue"] == "MEA"
        assert test_race["race_date"] == "2025-03-15"
        assert test_race["distance"] == "520m"
        assert test_race["grade"] == "Group 1"
        assert test_race["race_number"] == 8

        # Track condition should still be Unknown (not in headers for upcoming races)
        assert test_race["track_condition"] == "Unknown"

        # Cleanup
        os.remove(file_path)

    def test_multiple_csv_formats_compatibility(self, client, test_app):
        """Test that different CSV formats are handled correctly"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        csv_formats = [
            {
                "filename": "format1_spaces.csv",
                "content": """Race Name, Venue, Race Date, Distance
Test Race, WPK, 2025-02-01, 500m""",
            },
            {
                "filename": "format2_underscores.csv",
                "content": """Race_Name,Venue,Race_Date,Distance
Test_Race,SAN,2025-02-02,520m""",
            },
            {
                "filename": "format3_mixed.csv",
                "content": """RaceName,venue,racedate,Distance
TestRace,DAPT,2025-02-03,480m""",
            },
        ]

        # Create test files
        for fmt in csv_formats:
            file_path = os.path.join(upcoming_dir, fmt["filename"])
            with open(file_path, "w") as f:
                f.write(fmt["content"])

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        # Should handle all formats without errors
        assert response.status_code == 200
        assert data["success"] is True
        assert len(data["races"]) >= 3

        # Cleanup
        for fmt in csv_formats:
            file_path = os.path.join(upcoming_dir, fmt["filename"])
            if os.path.exists(file_path):
                os.remove(file_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
