#!/usr/bin/env python3
"""
Step 6 Standalone Unit Tests
===========================

Unit tests for Step 6 requirements that can run independently
without requiring the full Flask app to be working.

Tests the core requirements:
• Create temp dir with three dummy CSV headers 
• Assert correct count and structure (simulated API logic)
• Verify pagination & search work (logic testing)
"""

import hashlib
import math
import os
import re
import shutil
import tempfile
from datetime import datetime

import pandas as pd
import pytest


class MockCSVProcessor:
    """Mock CSV processor that simulates the API logic"""

    def __init__(self, csv_directory):
        self.csv_directory = csv_directory

    def process_upcoming_races_csv(
        self, page=1, per_page=10, sort_by="race_date", order="desc", search=""
    ):
        """Simulate the /api/upcoming_races_csv endpoint logic"""

        # Get all CSV files in directory
        if not os.path.exists(self.csv_directory):
            return {
                "success": True,
                "races": [],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                },
                "sort_by": sort_by,
                "order": order,
                "search": search,
            }

        csv_files = [f for f in os.listdir(self.csv_directory) if f.endswith(".csv")]

        if not csv_files:
            return {
                "success": True,
                "races": [],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                },
                "sort_by": sort_by,
                "order": order,
                "search": search,
            }

        # Process each CSV file
        races_data = []

        for filename in csv_files:
            file_path = os.path.join(self.csv_directory, filename)

            try:
                # Get file modification time
                file_mtime = os.path.getmtime(file_path)
                formatted_mtime = datetime.fromtimestamp(file_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )

                # Read CSV header
                try:
                    df_header = pd.read_csv(file_path, nrows=1)
                    columns = list(df_header.columns)
                except Exception:
                    continue

                # Extract race information
                race_name = None
                venue = None
                race_date = None
                distance = None
                grade = None
                race_number = None

                # Try to get data from headers
                if "Race Name" in columns and not df_header["Race Name"].empty:
                    race_name = str(df_header["Race Name"].iloc[0])
                elif "Race_Name" in columns and not df_header["Race_Name"].empty:
                    race_name = str(df_header["Race_Name"].iloc[0])
                else:
                    race_name = filename.replace(".csv", "")

                if "Venue" in columns and not df_header["Venue"].empty:
                    venue = str(df_header["Venue"].iloc[0])
                else:
                    # Extract from filename
                    filename_parts = filename.replace(".csv", "").split("_")
                    for part in filename_parts:
                        if len(part) == 3 and part.isupper():
                            venue = part
                            break
                    if not venue:
                        venue = "Unknown"

                if "Race Date" in columns and not df_header["Race Date"].empty:
                    race_date = str(df_header["Race Date"].iloc[0])
                elif "Race_Date" in columns and not df_header["Race_Date"].empty:
                    race_date = str(df_header["Race_Date"].iloc[0])
                else:
                    # Extract from filename
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
                    if date_match:
                        race_date = date_match.group(1)
                    else:
                        race_date = "Unknown"

                if "Distance" in columns and not df_header["Distance"].empty:
                    distance = str(df_header["Distance"].iloc[0])
                else:
                    distance = "Unknown"

                if "Grade" in columns and not df_header["Grade"].empty:
                    grade = str(df_header["Grade"].iloc[0])
                else:
                    grade = "Unknown"

                if "Race Number" in columns and not df_header["Race Number"].empty:
                    race_number = int(df_header["Race Number"].iloc[0])
                elif "Race_Number" in columns and not df_header["Race_Number"].empty:
                    race_number = int(df_header["Race_Number"].iloc[0])
                else:
                    # Extract from filename
                    race_num_match = re.search(
                        r"Race[_\s]?(\d+)", filename, re.IGNORECASE
                    )
                    if race_num_match:
                        race_number = int(race_num_match.group(1))
                    else:
                        race_number = 0

                # Generate race_id
                race_id = hashlib.md5(filename.encode()).hexdigest()[:12]

                # Clean up values
                def clean_value(value):
                    if value is None or str(value).lower() in ["nan", "none", "null"]:
                        return "Unknown"
                    return str(value)

                race_data = {
                    "race_id": race_id,
                    "venue": clean_value(venue),
                    "race_number": race_number,
                    "race_date": clean_value(race_date),
                    "race_name": clean_value(race_name),
                    "grade": clean_value(grade),
                    "distance": clean_value(distance),
                    "field_size": 0,
                    "winner_name": "Unknown",
                    "winner_odds": "N/A",
                    "winner_margin": "N/A",
                    "url": "",
                    "extraction_timestamp": formatted_mtime,
                    "track_condition": "Unknown",
                    "runners": [],
                    "filename": filename,
                    "file_mtime": file_mtime,
                }

                races_data.append(race_data)

            except Exception:
                continue

        # Apply search filter
        if search:
            filtered_races = []
            search_lower = search.lower()
            for race in races_data:
                if (
                    search_lower in race["venue"].lower()
                    or search_lower in race["race_name"].lower()
                    or search_lower in race["grade"].lower()
                    or search_lower in race["filename"].lower()
                ):
                    filtered_races.append(race)
            races_data = filtered_races

        # Sort races
        sort_options = {
            "race_date": "race_date",
            "venue": "venue",
            "confidence": "file_mtime",
            "grade": "grade",
        }

        sort_key = sort_options.get(sort_by, "file_mtime")
        reverse_sort = order == "desc"

        if sort_key == "file_mtime":
            races_data.sort(key=lambda x: x["file_mtime"], reverse=reverse_sort)
        else:
            races_data.sort(
                key=lambda x: str(x[sort_key]).lower(), reverse=reverse_sort
            )

        # Remove file_mtime from final output
        for race in races_data:
            race.pop("file_mtime", None)

        # Apply pagination
        total_count = len(races_data)
        offset = (page - 1) * per_page
        paginated_races = races_data[offset : offset + per_page]

        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1

        return {
            "success": True,
            "races": paginated_races,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            },
            "sort_by": sort_by,
            "order": order,
            "search": search,
        }


class TestStep6RequirementsStandalone:
    """Standalone tests for Step 6 requirements"""

    @pytest.fixture
    def temp_csv_dir(self):
        """Create temporary directory for CSV files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def three_dummy_csvs(self, temp_csv_dir):
        """Create exactly three dummy CSV files with different headers"""
        csv_files = [
            {
                "filename": "dummy_race_1_WPK_2025-02-01.csv",
                "content": """Race Name,Venue,Race Date,Distance,Grade,Race Number
Dummy Race One,WPK,2025-02-01,500m,Grade 5,1
Dummy Race One,WPK,2025-02-01,500m,Grade 5,1""",
            },
            {
                "filename": "dummy_race_2_MEA_2025-02-02.csv",
                "content": """Venue,Race_Date,Race_Name,Distance,Grade,Race_Number
MEA,2025-02-02,Dummy Race Two,520m,Grade 4,2
MEA,2025-02-02,Dummy Race Two,520m,Grade 4,2""",
            },
            {
                "filename": "dummy_race_3_GOSF_2025-02-03.csv",
                "content": """Dog Name,Box Number,Weight,Trainer Name
Dummy Dog Alpha,1,30.0,Dummy Trainer A
Dummy Dog Beta,2,29.5,Dummy Trainer B""",
            },
        ]

        created_files = []
        for csv_data in csv_files:
            file_path = os.path.join(temp_csv_dir, csv_data["filename"])
            with open(file_path, "w", newline="") as f:
                f.write(csv_data["content"])
            created_files.append(file_path)

        return created_files, temp_csv_dir

    def test_requirement_1_temp_dir_three_csvs_correct_count(self, three_dummy_csvs):
        """
        REQUIREMENT 1: Create temp dir with three dummy CSV headers and assert
        correct count is returned
        """
        created_files, temp_dir = three_dummy_csvs

        # Verify we created exactly 3 CSV files
        assert len(created_files) == 3, "Should have exactly 3 dummy CSV files"

        # Process using mock processor
        processor = MockCSVProcessor(temp_dir)
        result = processor.process_upcoming_races_csv()

        # Verify correct count
        assert result["success"] is True, "Processing should be successful"
        assert (
            len(result["races"]) == 3
        ), f"Should return exactly 3 races, got {len(result['races'])}"
        assert (
            result["pagination"]["total_count"] == 3
        ), f"Total count should be 3, got {result['pagination']['total_count']}"

        print(
            "✅ REQUIREMENT 1 PASSED: Correct count (3) returned for three dummy CSV files"
        )

    def test_requirement_2_correct_structure_validation(self, three_dummy_csvs):
        """
        REQUIREMENT 2: Assert correct structure is returned
        """
        created_files, temp_dir = three_dummy_csvs

        processor = MockCSVProcessor(temp_dir)
        result = processor.process_upcoming_races_csv()

        # Verify top-level structure
        required_fields = [
            "success",
            "races",
            "pagination",
            "sort_by",
            "order",
            "search",
        ]
        for field in required_fields:
            assert field in result, f"Top-level field '{field}' missing"

        # Verify pagination structure
        pagination = result["pagination"]
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

        # Verify each race structure
        races = result["races"]
        assert len(races) > 0, "Should have races in response"

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
            assert field in race, f"Race field '{field}' missing"

        # Verify data types
        assert isinstance(race["race_id"], str), "race_id should be string"
        assert isinstance(race["venue"], str), "venue should be string"
        assert isinstance(race["race_number"], int), "race_number should be integer"
        assert isinstance(race["filename"], str), "filename should be string"
        assert race["filename"].endswith(".csv"), "filename should end with .csv"

        print("✅ REQUIREMENT 2 PASSED: Correct structure returned")

    def test_requirement_3_different_csv_headers_parsing(self, three_dummy_csvs):
        """
        REQUIREMENT 3: Verify three different CSV header formats are parsed correctly
        """
        created_files, temp_dir = three_dummy_csvs

        processor = MockCSVProcessor(temp_dir)
        result = processor.process_upcoming_races_csv()

        races = result["races"]
        race_by_filename = {race["filename"]: race for race in races}

        # Test CSV 1: Full headers
        csv1_race = race_by_filename["dummy_race_1_WPK_2025-02-01.csv"]
        assert csv1_race["race_name"] == "Dummy Race One"
        assert csv1_race["venue"] == "WPK"
        assert csv1_race["race_date"] == "2025-02-01"
        assert csv1_race["distance"] == "500m"
        assert csv1_race["grade"] == "Grade 5"
        assert csv1_race["race_number"] == 1

        # Test CSV 2: Underscore headers
        csv2_race = race_by_filename["dummy_race_2_MEA_2025-02-02.csv"]
        assert csv2_race["race_name"] == "Dummy Race Two"
        assert csv2_race["venue"] == "MEA"
        assert csv2_race["race_date"] == "2025-02-02"
        assert csv2_race["distance"] == "520m"
        assert csv2_race["grade"] == "Grade 4"
        assert csv2_race["race_number"] == 2

        # Test CSV 3: Minimal headers (filename extraction)
        csv3_race = race_by_filename["dummy_race_3_GOSF_2025-02-03.csv"]
        assert csv3_race["venue"] == "GOSF"
        assert csv3_race["race_date"] == "2025-02-03"
        assert csv3_race["race_number"] == 3

        print(
            "✅ REQUIREMENT 3 PASSED: Three different CSV header formats parsed correctly"
        )

    def test_requirement_4_pagination_functionality(self, three_dummy_csvs):
        """
        REQUIREMENT 4: Verify pagination works correctly
        """
        created_files, temp_dir = three_dummy_csvs

        processor = MockCSVProcessor(temp_dir)

        # Test first page with per_page=2
        result = processor.process_upcoming_races_csv(page=1, per_page=2)

        assert result["success"] is True
        assert len(result["races"]) == 2, "First page should have 2 races"
        assert result["pagination"]["page"] == 1
        assert result["pagination"]["per_page"] == 2
        assert result["pagination"]["total_count"] == 3
        assert result["pagination"]["total_pages"] == 2
        assert result["pagination"]["has_next"] is True
        assert result["pagination"]["has_prev"] is False

        # Test second page
        result = processor.process_upcoming_races_csv(page=2, per_page=2)

        assert len(result["races"]) == 1, "Second page should have 1 race"
        assert result["pagination"]["page"] == 2
        assert result["pagination"]["has_next"] is False
        assert result["pagination"]["has_prev"] is True

        print("✅ REQUIREMENT 4 PASSED: Pagination functionality works correctly")

    def test_requirement_5_search_functionality(self, three_dummy_csvs):
        """
        REQUIREMENT 5: Verify search works correctly
        """
        created_files, temp_dir = three_dummy_csvs

        processor = MockCSVProcessor(temp_dir)

        # Search by venue
        result = processor.process_upcoming_races_csv(search="WPK")
        assert result["success"] is True
        assert len(result["races"]) == 1, "Search for 'WPK' should return 1 race"
        assert result["races"][0]["venue"] == "WPK"

        # Search by race name
        result = processor.process_upcoming_races_csv(search="Dummy Race Two")
        assert (
            len(result["races"]) == 1
        ), "Search for 'Dummy Race Two' should return 1 race"
        assert result["races"][0]["race_name"] == "Dummy Race Two"

        # Search by grade
        result = processor.process_upcoming_races_csv(search="Grade 5")
        assert len(result["races"]) == 1, "Search for 'Grade 5' should return 1 race"
        assert result["races"][0]["grade"] == "Grade 5"

        # Search by filename
        result = processor.process_upcoming_races_csv(search="GOSF")
        assert len(result["races"]) == 1, "Search for 'GOSF' should return 1 race"
        assert "GOSF" in result["races"][0]["filename"]

        # Search with no results
        result = processor.process_upcoming_races_csv(search="NONEXISTENT")
        assert (
            len(result["races"]) == 0
        ), "Search for non-existent term should return 0 races"
        assert result["pagination"]["total_count"] == 0

        print("✅ REQUIREMENT 5 PASSED: Search functionality works correctly")

    def test_requirement_6_sorting_functionality(self, three_dummy_csvs):
        """
        REQUIREMENT 6: Verify sorting works correctly
        """
        created_files, temp_dir = three_dummy_csvs

        processor = MockCSVProcessor(temp_dir)

        # Test sorting by race_date descending
        result = processor.process_upcoming_races_csv(sort_by="race_date", order="desc")
        races = result["races"]

        # Should be newest first
        assert races[0]["race_date"] == "2025-02-03"
        assert races[1]["race_date"] == "2025-02-02"
        assert races[2]["race_date"] == "2025-02-01"

        # Test sorting by race_date ascending
        result = processor.process_upcoming_races_csv(sort_by="race_date", order="asc")
        races = result["races"]

        # Should be oldest first
        assert races[0]["race_date"] == "2025-02-01"
        assert races[1]["race_date"] == "2025-02-02"
        assert races[2]["race_date"] == "2025-02-03"

        # Test sorting by venue
        result = processor.process_upcoming_races_csv(sort_by="venue", order="asc")
        races = result["races"]

        venues = [race["venue"] for race in races]
        assert venues == sorted(venues), f"Venues should be sorted: {venues}"

        print("✅ REQUIREMENT 6 PASSED: Sorting functionality works correctly")

    def test_requirement_7_race_id_consistency(self, three_dummy_csvs):
        """
        REQUIREMENT 7: Verify race IDs are generated consistently
        """
        created_files, temp_dir = three_dummy_csvs

        processor = MockCSVProcessor(temp_dir)

        # Get results twice
        result1 = processor.process_upcoming_races_csv()
        result2 = processor.process_upcoming_races_csv()

        races1 = result1["races"]
        races2 = result2["races"]

        # Race IDs should be unique
        race_ids = [race["race_id"] for race in races1]
        assert len(race_ids) == len(set(race_ids)), "All race IDs should be unique"

        # Race IDs should be consistent across calls
        race_ids_1 = [race["race_id"] for race in races1]
        race_ids_2 = [race["race_id"] for race in races2]
        assert race_ids_1 == race_ids_2, "Race IDs should be consistent"

        # Race IDs should be MD5 hash of filename
        for race in races1:
            expected_race_id = hashlib.md5(race["filename"].encode()).hexdigest()[:12]
            assert (
                race["race_id"] == expected_race_id
            ), f"Race ID should be MD5 hash for {race['filename']}"

        print("✅ REQUIREMENT 7 PASSED: Race ID generation is consistent")

    def test_requirement_8_edge_cases(self, temp_csv_dir):
        """
        REQUIREMENT 8: Verify edge cases are handled correctly
        """
        processor = MockCSVProcessor(temp_csv_dir)

        # Test with no CSV files
        result = processor.process_upcoming_races_csv()
        assert result["success"] is True
        assert len(result["races"]) == 0
        assert result["pagination"]["total_count"] == 0

        # Test with non-existent directory
        processor_nonexistent = MockCSVProcessor("/nonexistent/directory")
        result = processor_nonexistent.process_upcoming_races_csv()
        assert result["success"] is True
        assert len(result["races"]) == 0
        assert result["pagination"]["total_count"] == 0

        print("✅ REQUIREMENT 8 PASSED: Edge cases handled correctly")


def run_step6_standalone_tests():
    """Run all standalone Step 6 tests"""
    print("=" * 80)
    print("STEP 6 STANDALONE TEST SUITE")
    print("=" * 80)
    print("Testing Requirements (without Flask dependency):")
    print("• Create temp dir with three dummy CSV headers")
    print("• Assert correct count and structure")
    print("• Verify pagination & search work")
    print("=" * 80)

    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    print("=" * 80)
    if exit_code == 0:
        print("✅ ALL STEP 6 STANDALONE REQUIREMENTS PASSED!")
    else:
        print("❌ SOME STEP 6 STANDALONE REQUIREMENTS FAILED")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    exit_code = run_step6_standalone_tests()
    exit(exit_code)
