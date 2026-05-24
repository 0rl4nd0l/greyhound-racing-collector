import os
import tempfile
import time
import unittest
from datetime import datetime

from utils.file_naming import (
    build_prediction_filename,
    get_filename_for_race_id,
    get_race_id_from_filename,
    is_valid_prediction_filename,
    parse_prediction_filename,
)


class TestPredictionFilename(unittest.TestCase):
    def test_get_filename_for_race_id(self):
        # Setup temporary directories
        with tempfile.TemporaryDirectory() as tmpdirname:
            upcoming_dir = os.path.join(tmpdirname, "upcoming")
            historical_dir = os.path.join(tmpdirname, "historical")
            os.mkdir(upcoming_dir)
            os.mkdir(historical_dir)

            # Create mock race files
            race_id = "Race_01_MAN_2025-01-15"
            upcoming_file = os.path.join(upcoming_dir, f"{race_id}.csv")
            historical_file = os.path.join(historical_dir, f"{race_id}_old.csv")

            with open(upcoming_file, "w") as f:
                f.write("Upcoming race data")
            with open(historical_file, "w") as f:
                f.write("Historical race data")

            # Test finding in the upcoming directory
            filename, full_path = get_filename_for_race_id(
                race_id, [upcoming_dir, historical_dir]
            )
            self.assertEqual(filename, f"{race_id}.csv")
            self.assertEqual(full_path, upcoming_file)

            # Test finding in the historical directory if not found in upcoming
            os.remove(upcoming_file)  # Remove from upcoming to test historical
            filename, full_path = get_filename_for_race_id(
                race_id, [upcoming_dir, historical_dir]
            )
            self.assertEqual(filename, f"{race_id}_old.csv")
            self.assertEqual(full_path, historical_file)

            # Test no file found
            os.remove(historical_file)
            filename, full_path = get_filename_for_race_id(
                race_id, [upcoming_dir, historical_dir]
            )
            self.assertIsNone(filename)
            self.assertIsNone(full_path)

    def test_race_id_extraction(self):
        test_cases = [
            ("Race_01_MAN_2025-01-15.csv", "01_MAN_2025-01-15"),
            ("Race 01 MAN 2025-01-15.csv", "01_MAN_2025-01-15"),
            ("Race-01-MAN-2025-01-15.csv", "01-MAN-2025-01-15"),  # dashes preserved
        ]

        for filename, expected_race_id in test_cases:
            extracted_id = get_race_id_from_filename(filename)
            self.assertEqual(extracted_id, expected_race_id, f"Failed for {filename}")

    def test_filename_uniqueness(self):
        race_id = "Race_99_TEST_2099-12-31"
        method = "supermodel"
        fnames = set()
        for _ in range(3):
            fname = build_prediction_filename(race_id, method=method)
            self.assertNotIn(fname, fnames)
            fnames.add(fname)
            time.sleep(1)  # force timestamp difference

    def test_filename_parseability(self):
        ts = datetime(2033, 7, 24, 15, 16, 55)
        race_id = "Race_01_HEA_2033-07-24"
        method = "v3"
        filename = build_prediction_filename(race_id, ts, method)
        self.assertTrue(is_valid_prediction_filename(filename))
        parsed = parse_prediction_filename(filename)
        self.assertTrue(parsed["is_valid"])
        self.assertEqual(parsed["race_id"], race_id)
        self.assertEqual(parsed["method"], method)
        self.assertEqual(parsed["timestamp"], ts)

    def test_bad_filenames(self):
        bad_names = [
            "bad_prediction.json",
            "prediction__missing_method_20230101_120000.json",
            "prediction_Race_XYZ_v3_.json",
            "prediction_Race__v3_20220101_010101.json",
            "not_a_predictionfile.json",
        ]
        for fname in bad_names:
            is_valid = is_valid_prediction_filename(fname)
            if is_valid:
                print(f"ERROR: {fname} was considered valid but should be invalid")
            self.assertFalse(
                is_valid, f"Filename {fname} should be invalid but was considered valid"
            )
            parsed = parse_prediction_filename(fname)
            self.assertFalse(
                parsed["is_valid"], f"Parsed result for {fname} should be invalid"
            )


class TestFilenameResolution(unittest.TestCase):
    """Test cases for filename resolution utilities."""

    def test_get_filename_for_race_id_default_paths(self):
        """Test get_filename_for_race_id with default paths."""
        # Test with non-existent race_id
        filename, full_path = get_filename_for_race_id("NonExistent_Race_ID")
        self.assertIsNone(filename)
        self.assertIsNone(full_path)

    def test_get_filename_for_race_id_exact_match(self):
        """Test exact filename matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            race_id = "01_MAN_2025-01-15"
            test_file = os.path.join(tmpdir, f"{race_id}.csv")
            with open(test_file, "w") as f:
                f.write("test data")

            # Test exact match
            filename, full_path = get_filename_for_race_id(race_id, [tmpdir])
            self.assertEqual(filename, f"{race_id}.csv")
            self.assertEqual(full_path, test_file)

    def test_get_filename_for_race_id_pattern_matching(self):
        """Test various filename pattern matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            race_id = "01_MAN_2025-01-15"

            # Create file with "Race" prefix
            test_file = os.path.join(tmpdir, f"Race_{race_id}.csv")
            with open(test_file, "w") as f:
                f.write("test data")

            # Test pattern matching
            filename, full_path = get_filename_for_race_id(race_id, [tmpdir])
            self.assertEqual(filename, f"Race_{race_id}.csv")
            self.assertEqual(full_path, test_file)

    def test_get_filename_for_race_id_partial_match(self):
        """Test partial matching when exact patterns don't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            race_id = "MAN"

            # Create file that contains the race_id but doesn't match exact patterns
            test_file = os.path.join(tmpdir, "Some_MAN_Race_File.csv")
            with open(test_file, "w") as f:
                f.write("test data")

            # Test partial matching
            filename, full_path = get_filename_for_race_id(race_id, [tmpdir])
            self.assertEqual(filename, "Some_MAN_Race_File.csv")
            self.assertEqual(full_path, test_file)

    def test_get_race_id_from_filename_various_formats(self):
        """Test race_id extraction from various filename formats."""
        test_cases = [
            ("Race_01_MAN_2025-01-15.csv", "01_MAN_2025-01-15"),
            ("Race 05 GEE 2025-07-30.csv", "05_GEE_2025-07-30"),
            ("Race-10-ALB-2025-12-25.csv", "10-ALB-2025-12-25"),
            ("01_MAN_2025-01-15.csv", "01_MAN_2025-01-15"),
            ("/path/to/Race_01_MAN_2025-01-15.csv", "01_MAN_2025-01-15"),
        ]

        for filename, expected_race_id in test_cases:
            with self.subTest(filename=filename):
                result = get_race_id_from_filename(filename)
                self.assertEqual(result, expected_race_id)

    def test_get_filename_for_race_id_directory_priority(self):
        """Test that directories are searched in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = os.path.join(tmpdir, "dir1")
            dir2 = os.path.join(tmpdir, "dir2")
            os.makedirs(dir1)
            os.makedirs(dir2)

            race_id = "test_race"

            # Create files in both directories
            file1 = os.path.join(dir1, f"{race_id}.csv")
            file2 = os.path.join(dir2, f"{race_id}.csv")

            with open(file1, "w") as f:
                f.write("dir1 data")
            with open(file2, "w") as f:
                f.write("dir2 data")

            # Should find in first directory
            filename, full_path = get_filename_for_race_id(race_id, [dir1, dir2])
            self.assertEqual(full_path, file1)

            # Should find in second directory if first is searched second
            filename, full_path = get_filename_for_race_id(race_id, [dir2, dir1])
            self.assertEqual(full_path, file2)

    def test_get_filename_for_race_id_nonexistent_directory(self):
        """Test behavior with non-existent directories."""
        nonexistent_dir = "/this/directory/does/not/exist"
        filename, full_path = get_filename_for_race_id("test_race", [nonexistent_dir])
        self.assertIsNone(filename)
        self.assertIsNone(full_path)


if __name__ == "__main__":
    unittest.main()
