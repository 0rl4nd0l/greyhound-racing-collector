import unittest
from datetime import datetime, timedelta
from utils.file_naming import build_prediction_filename, parse_prediction_filename, is_valid_prediction_filename
import time

class TestPredictionFilename(unittest.TestCase):
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
        self.assertTrue(parsed['is_valid'])
        self.assertEqual(parsed['race_id'], race_id)
        self.assertEqual(parsed['method'], method)
        self.assertEqual(parsed['timestamp'], ts)

    def test_bad_filenames(self):
        bad_names = [
            "bad_prediction.json",
            "prediction__missing_method_20230101_120000.json",
            "prediction_Race_XYZ_v3_.json",
            "prediction_Race__v3_20220101_010101.json",
            "not_a_predictionfile.json"
        ]
        for fname in bad_names:
            is_valid = is_valid_prediction_filename(fname)
            if is_valid:
                print(f"ERROR: {fname} was considered valid but should be invalid")
            self.assertFalse(is_valid, f"Filename {fname} should be invalid but was considered valid")
            parsed = parse_prediction_filename(fname)
            self.assertFalse(parsed['is_valid'], f"Parsed result for {fname} should be invalid")

if __name__ == "__main__":
    unittest.main()

