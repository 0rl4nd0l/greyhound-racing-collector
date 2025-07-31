import sqlite3
import unittest

class TestDatabaseIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.connection = sqlite3.connect('greyhound_racing_data.db')

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def test_gpt_analysis_table(self):
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA table_info(gpt_analysis)")
        columns = [info[1] for info in cursor.fetchall()]
        required_columns = ['id', 'race_id', 'analysis_type', 'analysis_data']
        for col in required_columns:
            self.assertIn(col, columns, f"{col} is missing from gpt_analysis")

    def test_race_metadata_table(self):
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA table_info(race_metadata)")
        columns = [info[1] for info in cursor.fetchall()]
        required_columns = ['id', 'race_id', 'venue', 'race_number', 'race_date']
        for col in required_columns:
            self.assertIn(col, columns, f"{col} is missing from race_metadata")

    def test_dog_race_data_table(self):
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA table_info(dog_race_data)")
        columns = [info[1] for info in cursor.fetchall()]
        required_columns = ['id', 'race_id', 'dog_name', 'finish_position']
        for col in required_columns:
            self.assertIn(col, columns, f"{col} is missing from dog_race_data")

if __name__ == "__main__":
    unittest.main()
