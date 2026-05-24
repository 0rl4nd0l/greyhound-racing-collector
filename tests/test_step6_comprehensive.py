#!/usr/bin/env python3
"""
Step 6 Comprehensive Test Suite
==============================

Comprehensive test runner that validates all Step 6 requirements:
• Pytest: create temp dir with three dummy CSV headers and assert `/api/upcoming_races_csv` returns correct count and structure
• Selenium/Playwright (optional): load interactive-races page, ensure rows appear after download
• Verify pagination & search work

This test suite combines unit tests, integration tests, and end-to-end verification.
"""

import hashlib
import os
import shutil
import sys
import tempfile

import pytest

# Import test fixtures and utilities


class TestStep6Requirements:
    """Main test class that validates all Step 6 requirements"""

    @pytest.fixture
    def create_three_dummy_csvs(self, test_app):
        """Create temporary directory with exactly three dummy CSV files with different headers"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create exactly three CSV files with different header structures as required
        csv_files = [
            {
                "filename": "dummy_race_1_WPK_2025-02-01.csv",
                "headers": [
                    "Race Name",
                    "Venue",
                    "Race Date",
                    "Distance",
                    "Grade",
                    "Race Number",
                ],
                "content": """Race Name,Venue,Race Date,Distance,Grade,Race Number
Dummy Race One,WPK,2025-02-01,500m,Grade 5,1
Dummy Race One,WPK,2025-02-01,500m,Grade 5,1""",
            },
            {
                "filename": "dummy_race_2_MEA_2025-02-02.csv",
                "headers": [
                    "Venue",
                    "Race_Date",
                    "Race_Name",
                    "Distance",
                    "Grade",
                    "Race_Number",
                ],
                "content": """Venue,Race_Date,Race_Name,Distance,Grade,Race_Number
MEA,2025-02-02,Dummy Race Two,520m,Grade 4,2
MEA,2025-02-02,Dummy Race Two,520m,Grade 4,2""",
            },
            {
                "filename": "dummy_race_3_GOSF_2025-02-03.csv",
                "headers": [
                    "Dog Name",
                    "Box Number",
                    "Weight",
                    "Trainer Name",
                ],  # Minimal headers
                "content": """Dog Name,Box Number,Weight,Trainer Name
Dummy Dog Alpha,1,30.0,Dummy Trainer A
Dummy Dog Beta,2,29.5,Dummy Trainer B
Dummy Dog Gamma,3,30.8,Dummy Trainer C""",
            },
        ]

        # Write the three CSV files
        created_files = []
        for csv_data in csv_files:
            file_path = os.path.join(upcoming_dir, csv_data["filename"])
            with open(file_path, "w", newline="") as f:
                f.write(csv_data["content"])
            created_files.append(
                {
                    "path": file_path,
                    "filename": csv_data["filename"],
                    "headers": csv_data["headers"],
                }
            )

        yield created_files

        # Cleanup handled by test_app fixture

    def test_requirement_1_temp_dir_three_csvs_correct_count(
        self, client, create_three_dummy_csvs
    ):
        """
        REQUIREMENT 1: Create temp dir with three dummy CSV headers and assert
        `/api/upcoming_races_csv` returns correct count
        """
        csv_files = create_three_dummy_csvs

        # Verify we created exactly 3 CSV files
        assert len(csv_files) == 3, "Should have exactly 3 dummy CSV files"

        # Call the API endpoint
        response = client.get("/api/upcoming_races_csv")
        assert response.status_code == 200, "API endpoint should return 200 OK"

        data = response.get_json()
        assert data is not None, "API should return JSON data"
        assert data["success"] is True, "API should return success=True"

        # VERIFY CORRECT COUNT: Should return exactly 3 races
        assert (
            len(data["races"]) == 3
        ), f"API should return exactly 3 races, got {len(data['races'])}"
        assert (
            data["pagination"]["total_count"] == 3
        ), f"Total count should be 3, got {data['pagination']['total_count']}"

        print(
            "✅ REQUIREMENT 1 PASSED: API returns correct count (3) for three dummy CSV files"
        )

    def test_requirement_2_api_structure_validation(
        self, client, create_three_dummy_csvs
    ):
        """
        REQUIREMENT 2: Assert `/api/upcoming_races_csv` returns correct structure
        """
        csv_files = create_three_dummy_csvs

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        # Verify top-level structure
        required_top_level_fields = [
            "success",
            "races",
            "pagination",
            "sort_by",
            "order",
            "search",
        ]
        for field in required_top_level_fields:
            assert field in data, f"Top-level field '{field}' missing from API response"

        # Verify pagination structure
        pagination = data["pagination"]
        required_pagination_fields = [
            "page",
            "per_page",
            "total_count",
            "total_pages",
            "has_next",
            "has_prev",
        ]
        for field in required_pagination_fields:
            assert field in pagination, f"Pagination field '{field}' missing"

        # Verify each race has correct structure
        races = data["races"]
        assert len(races) > 0, "Should have at least one race in response"

        race = races[0]
        required_race_fields = [
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

        for field in required_race_fields:
            assert field in race, f"Race field '{field}' missing from race data"

        # Verify data types
        assert isinstance(race["race_id"], str), "race_id should be string"
        assert isinstance(race["venue"], str), "venue should be string"
        assert isinstance(race["race_number"], int), "race_number should be integer"
        assert isinstance(race["race_date"], str), "race_date should be string"
        assert isinstance(race["race_name"], str), "race_name should be string"
        assert isinstance(race["grade"], str), "grade should be string"
        assert isinstance(race["distance"], str), "distance should be string"
        assert isinstance(race["field_size"], int), "field_size should be integer"
        assert isinstance(race["runners"], list), "runners should be list"
        assert isinstance(race["filename"], str), "filename should be string"
        assert race["filename"].endswith(".csv"), "filename should end with .csv"

        # Verify upcoming race specific values
        assert (
            race["winner_name"] == "Unknown"
        ), "Upcoming races should have winner_name='Unknown'"
        assert (
            race["winner_odds"] == "N/A"
        ), "Upcoming races should have winner_odds='N/A'"
        assert (
            race["winner_margin"] == "N/A"
        ), "Upcoming races should have winner_margin='N/A'"
        assert race["url"] == "", "Upcoming races should have empty URL"
        assert (
            race["track_condition"] == "Unknown"
        ), "Upcoming races should have track_condition='Unknown'"

        print("✅ REQUIREMENT 2 PASSED: API returns correct structure for all races")

    def test_requirement_3_different_csv_headers_parsing(
        self, client, create_three_dummy_csvs
    ):
        """
        REQUIREMENT 3: Verify that three dummy CSV files with different headers are parsed correctly
        """
        csv_files = create_three_dummy_csvs

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        races = data["races"]
        race_by_filename = {race["filename"]: race for race in races}

        # Test CSV 1: Full headers (Race Name, Venue, Race Date, Distance, Grade, Race Number)
        csv1_race = race_by_filename["dummy_race_1_WPK_2025-02-01.csv"]
        assert (
            csv1_race["race_name"] == "Dummy Race One"
        ), "CSV1: Race name should be parsed from headers"
        assert csv1_race["venue"] == "WPK", "CSV1: Venue should be parsed from headers"
        assert (
            csv1_race["race_date"] == "2025-02-01"
        ), "CSV1: Race date should be parsed from headers"
        assert (
            csv1_race["distance"] == "500m"
        ), "CSV1: Distance should be parsed from headers"
        assert (
            csv1_race["grade"] == "Grade 5"
        ), "CSV1: Grade should be parsed from headers"
        assert (
            csv1_race["race_number"] == 1
        ), "CSV1: Race number should be parsed from headers"

        # Test CSV 2: Underscore headers (Venue, Race_Date, Race_Name, Distance, Grade, Race_Number)
        csv2_race = race_by_filename["dummy_race_2_MEA_2025-02-02.csv"]
        assert (
            csv2_race["race_name"] == "Dummy Race Two"
        ), "CSV2: Race name should be parsed from underscore headers"
        assert (
            csv2_race["venue"] == "MEA"
        ), "CSV2: Venue should be parsed from underscore headers"
        assert (
            csv2_race["race_date"] == "2025-02-02"
        ), "CSV2: Race date should be parsed from underscore headers"
        assert (
            csv2_race["distance"] == "520m"
        ), "CSV2: Distance should be parsed from underscore headers"
        assert (
            csv2_race["grade"] == "Grade 4"
        ), "CSV2: Grade should be parsed from underscore headers"
        assert (
            csv2_race["race_number"] == 2
        ), "CSV2: Race number should be parsed from underscore headers"

        # Test CSV 3: Minimal headers (should extract from filename)
        csv3_race = race_by_filename["dummy_race_3_GOSF_2025-02-03.csv"]
        assert (
            csv3_race["venue"] == "GOSF"
        ), "CSV3: Venue should be extracted from filename"
        assert (
            csv3_race["race_date"] == "2025-02-03"
        ), "CSV3: Race date should be extracted from filename"
        assert (
            csv3_race["race_number"] == 3
        ), "CSV3: Race number should be extracted from filename"
        # Race name should fallback to filename-based name
        assert (
            "dummy_race_3" in csv3_race["race_name"].lower()
            or csv3_race["race_name"] == "dummy_race_3_GOSF_2025-02-03"
        ), "CSV3: Race name should be derived from filename"

        print(
            "✅ REQUIREMENT 3 PASSED: Three different CSV header formats parsed correctly"
        )

    def test_requirement_4_pagination_functionality(
        self, client, create_three_dummy_csvs
    ):
        """
        REQUIREMENT 4: Verify pagination works correctly
        """
        csv_files = create_three_dummy_csvs

        # Test pagination with per_page=2 (should create 2 pages for 3 items)
        response = client.get("/api/upcoming_races_csv?page=1&per_page=2")
        assert response.status_code == 200

        data = response.get_json()
        assert data["success"] is True

        # Verify first page
        assert len(data["races"]) == 2, "First page should have 2 races"
        assert data["pagination"]["page"] == 1, "Should be on page 1"
        assert data["pagination"]["per_page"] == 2, "Per page should be 2"
        assert data["pagination"]["total_count"] == 3, "Total count should be 3"
        assert data["pagination"]["total_pages"] == 2, "Should have 2 total pages"
        assert data["pagination"]["has_next"] is True, "Should have next page"
        assert data["pagination"]["has_prev"] is False, "Should not have previous page"

        # Test second page
        response = client.get("/api/upcoming_races_csv?page=2&per_page=2")
        data = response.get_json()

        assert len(data["races"]) == 1, "Second page should have 1 race"
        assert data["pagination"]["page"] == 2, "Should be on page 2"
        assert data["pagination"]["has_next"] is False, "Should not have next page"
        assert data["pagination"]["has_prev"] is True, "Should have previous page"

        # Test invalid pagination parameters
        response = client.get("/api/upcoming_races_csv?page=0")
        assert response.status_code == 400, "Page 0 should return 400 error"

        response = client.get("/api/upcoming_races_csv?per_page=0")
        assert response.status_code == 400, "Per_page 0 should return 400 error"

        print("✅ REQUIREMENT 4 PASSED: Pagination functionality works correctly")

    def test_requirement_5_search_functionality(self, client, create_three_dummy_csvs):
        """
        REQUIREMENT 5: Verify search works correctly
        """
        csv_files = create_three_dummy_csvs

        # Search by venue
        response = client.get("/api/upcoming_races_csv?search=WPK")
        data = response.get_json()
        assert data["success"] is True
        assert len(data["races"]) == 1, "Search for 'WPK' should return 1 race"
        assert data["races"][0]["venue"] == "WPK", "Found race should have venue 'WPK'"

        # Search by race name
        response = client.get("/api/upcoming_races_csv?search=Dummy Race Two")
        data = response.get_json()
        assert (
            len(data["races"]) == 1
        ), "Search for 'Dummy Race Two' should return 1 race"
        assert (
            data["races"][0]["race_name"] == "Dummy Race Two"
        ), "Found race should have correct race name"

        # Search by grade
        response = client.get("/api/upcoming_races_csv?search=Grade 5")
        data = response.get_json()
        assert len(data["races"]) == 1, "Search for 'Grade 5' should return 1 race"
        assert data["races"][0]["grade"] == "Grade 5", "Found race should have Grade 5"

        # Search by filename
        response = client.get("/api/upcoming_races_csv?search=GOSF")
        data = response.get_json()
        assert len(data["races"]) == 1, "Search for 'GOSF' should return 1 race"
        assert (
            "GOSF" in data["races"][0]["filename"]
        ), "Found race filename should contain 'GOSF'"

        # Search with no results
        response = client.get("/api/upcoming_races_csv?search=NONEXISTENT_VENUE")
        data = response.get_json()
        assert (
            len(data["races"]) == 0
        ), "Search for non-existent term should return 0 races"
        assert (
            data["pagination"]["total_count"] == 0
        ), "Total count should be 0 for no results"

        # Search is case-insensitive
        response = client.get("/api/upcoming_races_csv?search=wpk")
        data = response.get_json()
        assert len(data["races"]) == 1, "Search should be case-insensitive"

        print("✅ REQUIREMENT 5 PASSED: Search functionality works correctly")

    def test_requirement_6_sorting_functionality(self, client, create_three_dummy_csvs):
        """
        REQUIREMENT 6: Verify sorting works correctly
        """
        csv_files = create_three_dummy_csvs

        # Test sorting by race_date descending (newest first)
        response = client.get("/api/upcoming_races_csv?sort_by=race_date&order=desc")
        data = response.get_json()
        races = data["races"]

        # Should be sorted: 2025-02-03, 2025-02-02, 2025-02-01
        assert races[0]["race_date"] == "2025-02-03", "First race should be newest date"
        assert (
            races[1]["race_date"] == "2025-02-02"
        ), "Second race should be middle date"
        assert races[2]["race_date"] == "2025-02-01", "Third race should be oldest date"

        # Test sorting by race_date ascending (oldest first)
        response = client.get("/api/upcoming_races_csv?sort_by=race_date&order=asc")
        data = response.get_json()
        races = data["races"]

        # Should be sorted: 2025-02-01, 2025-02-02, 2025-02-03
        assert races[0]["race_date"] == "2025-02-01", "First race should be oldest date"
        assert (
            races[1]["race_date"] == "2025-02-02"
        ), "Second race should be middle date"
        assert races[2]["race_date"] == "2025-02-03", "Third race should be newest date"

        # Test sorting by venue ascending
        response = client.get("/api/upcoming_races_csv?sort_by=venue&order=asc")
        data = response.get_json()
        races = data["races"]

        venues = [race["venue"] for race in races]
        assert venues == sorted(
            venues
        ), f"Venues should be sorted alphabetically: {venues}"

        print("✅ REQUIREMENT 6 PASSED: Sorting functionality works correctly")

    def test_requirement_7_race_id_generation_consistency(
        self, client, create_three_dummy_csvs
    ):
        """
        REQUIREMENT 7: Verify race IDs are generated consistently
        """
        csv_files = create_three_dummy_csvs

        response = client.get("/api/upcoming_races_csv")
        data = response.get_json()

        races = data["races"]
        race_ids = [race["race_id"] for race in races]

        # All race IDs should be unique
        assert len(race_ids) == len(set(race_ids)), "All race IDs should be unique"

        # Race IDs should be consistent (MD5 hash of filename)
        for race in races:
            expected_race_id = hashlib.md5(race["filename"].encode()).hexdigest()[:12]
            assert (
                race["race_id"] == expected_race_id
            ), f"Race ID should be MD5 hash of filename for {race['filename']}"

        # Call API again and verify race IDs are same (consistency test)
        response2 = client.get("/api/upcoming_races_csv")
        data2 = response2.get_json()
        races2 = data2["races"]

        race_ids_2 = [race["race_id"] for race in races2]
        assert race_ids == race_ids_2, "Race IDs should be consistent across API calls"

        print("✅ REQUIREMENT 7 PASSED: Race ID generation is consistent")

    def test_requirement_8_error_handling_edge_cases(self, client, test_app):
        """
        REQUIREMENT 8: Verify proper error handling for edge cases
        """

        # Test with no upcoming_races directory
        upcoming_dir = test_app.config["UPCOMING_DIR"]
        if os.path.exists(upcoming_dir):
            # Temporarily remove directory
            backup_dir = tempfile.mkdtemp()
            shutil.move(upcoming_dir, os.path.join(backup_dir, "upcoming_races"))

        response = client.get("/api/upcoming_races_csv")
        assert response.status_code == 200, "Should handle missing directory gracefully"
        data = response.get_json()
        assert data["success"] is True, "Should return success even with no directory"
        assert len(data["races"]) == 0, "Should return empty races list"
        assert data["pagination"]["total_count"] == 0, "Total count should be 0"

        # Restore directory
        if "backup_dir" in locals():
            shutil.move(os.path.join(backup_dir, "upcoming_races"), upcoming_dir)
            shutil.rmtree(backup_dir)
        else:
            os.makedirs(upcoming_dir, exist_ok=True)

        print("✅ REQUIREMENT 8 PASSED: Error handling works correctly")


class TestStep6IntegrationRequirements:
    """Integration test requirements for Step 6"""

    def test_api_integration_with_real_csv_data(self, client, test_app):
        """Test API with real CSV data structure"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create a more realistic CSV structure
        realistic_csv = """Race Name,Venue,Race Date,Distance,Grade,Race Number,Dog Name,Box,Weight,Trainer
Melbourne Cup Heat 1,MEA,2025-03-15,520m,Group 1,1,Lightning Bolt,1,32.5,John Smith
Melbourne Cup Heat 1,MEA,2025-03-15,520m,Group 1,1,Thunder Strike,2,31.8,Jane Doe
Melbourne Cup Heat 1,MEA,2025-03-15,520m,Group 1,1,Speed Demon,3,30.9,Bob Wilson
Melbourne Cup Heat 1,MEA,2025-03-15,520m,Group 1,1,Fast Track,4,32.1,Alice Brown
Melbourne Cup Heat 1,MEA,2025-03-15,520m,Group 1,1,Quick Silver,5,31.4,Charlie Davis
Melbourne Cup Heat 1,MEA,2025-03-15,520m,Group 1,1,Rapid Fire,6,30.6,Diana Evans"""

        realistic_file = os.path.join(
            upcoming_dir, "MEA_Melbourne_Cup_Heat_1_2025-03-15.csv"
        )
        with open(realistic_file, "w") as f:
            f.write(realistic_csv)

        try:
            response = client.get("/api/upcoming_races_csv")
            assert response.status_code == 200

            data = response.get_json()
            assert data["success"] is True
            assert len(data["races"]) >= 1

            # Find our realistic race
            race = None
            for r in data["races"]:
                if "Melbourne_Cup" in r["filename"]:
                    race = r
                    break

            assert race is not None, "Realistic race should be found"
            assert race["race_name"] == "Melbourne Cup Heat 1"
            assert race["venue"] == "MEA"
            assert race["race_date"] == "2025-03-15"
            assert race["distance"] == "520m"
            assert race["grade"] == "Group 1"
            assert race["race_number"] == 1

            print("✅ INTEGRATION TEST PASSED: Real CSV data processed correctly")

        finally:
            if os.path.exists(realistic_file):
                os.remove(realistic_file)

    def test_combined_pagination_and_search(self, client, test_app):
        """Test pagination and search working together"""
        upcoming_dir = test_app.config["UPCOMING_DIR"]

        # Create multiple CSV files for combined testing
        test_csvs = []
        for i in range(6):  # Create 6 races for pagination testing
            venue = ["WPK", "MEA", "GOSF", "DAPT", "SAN", "BAL"][i]
            csv_content = f"""Race Name,Venue,Race Date,Distance,Grade,Race Number
Test Race {i+1},{venue},2025-02-{i+1:02d},500m,Grade {(i%5)+1},{i+1}"""

            filename = f"Test_Race_{i+1}_{venue}_2025-02-{i+1:02d}.csv"
            filepath = os.path.join(upcoming_dir, filename)

            with open(filepath, "w") as f:
                f.write(csv_content)
            test_csvs.append(filepath)

        try:
            # Test search with pagination
            response = client.get(
                "/api/upcoming_races_csv?search=WPK&page=1&per_page=5"
            )
            data = response.get_json()

            assert data["success"] is True
            assert len(data["races"]) == 1  # Only one WPK race
            assert data["races"][0]["venue"] == "WPK"
            assert data["pagination"]["total_count"] == 1

            # Test pagination without search
            response = client.get("/api/upcoming_races_csv?page=1&per_page=3")
            data = response.get_json()

            assert len(data["races"]) == 3  # First page with 3 races
            assert (
                data["pagination"]["total_pages"] == 2
            )  # 6 races / 3 per page = 2 pages
            assert data["pagination"]["has_next"] is True

            print("✅ COMBINED TEST PASSED: Pagination and search work together")

        finally:
            # Cleanup
            for filepath in test_csvs:
                if os.path.exists(filepath):
                    os.remove(filepath)


def run_comprehensive_step6_tests():
    """Run all Step 6 tests and generate summary report"""
    print("=" * 80)
    print("STEP 6 COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing Requirements:")
    print("• Pytest: create temp dir with three dummy CSV headers")
    print("• Assert `/api/upcoming_races_csv` returns correct count and structure")
    print("• Verify pagination & search work")
    print("=" * 80)

    # Run the tests
    exit_code = pytest.main(
        [__file__, "-v", "--tb=short", "--maxfail=10", "--durations=10"]
    )

    print("=" * 80)
    if exit_code == 0:
        print("✅ ALL STEP 6 REQUIREMENTS PASSED!")
    else:
        print("❌ SOME STEP 6 REQUIREMENTS FAILED")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    exit_code = run_comprehensive_step6_tests()
    sys.exit(exit_code)
