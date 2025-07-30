
import sqlite3

DATABASE_PATH = 'greyhound_racing_data.db'

CRITICAL_TABLES = [
    'race_metadata',
    'dog_race_data',
    'dogs',
    'dog_performances',
    'live_odds',
    'odds_history',
    'predictions',
    'weather_data'
]

def run_integrity_test():
    """
    Verifies that critical tables return at least one row from a simple query.
    """
    print("Running database integrity tests...")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    all_tests_passed = True
    
    for table in CRITICAL_TABLES:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"✅ {table}: OK ({count} rows)")
            else:
                print(f"❌ {table}: FAILED (0 rows)")
                all_tests_passed = False
        except sqlite3.Error as e:
            print(f"❌ {table}: FAILED ({e})")
            all_tests_passed = False
            
    conn.close()
    
    if all_tests_passed:
        print("\n✅ All critical table integrity tests passed!")
    else:
        print("\n❌ Some integrity tests failed.")

if __name__ == '__main__':
    run_integrity_test()

