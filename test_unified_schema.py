#!/usr/bin/env python3
"""
Unit test to verify unified schema compatibility
"""

import sqlite3
import sys
import os

def test_database_connection():
    """Test basic database connection and table existence"""
    print("🔍 Testing database connection and schema...")
    
    try:
        conn = sqlite3.connect('greyhound_racing_data_test.db')
        cursor = conn.cursor()
        
        # Test main tables exist
        tables_to_check = ['race_metadata', 'dog_race_data', 'dogs']
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  ✅ {table}: {count} records")
        
        # Test join query (used by ML system)
        cursor.execute("""
            SELECT 
                drd.dog_name,
                drd.finish_position,
                rm.venue,
                rm.race_date,
                rm.distance
            FROM dog_race_data drd
            LEFT JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
                AND drd.finish_position != '' 
                AND drd.finish_position != 'N/A'
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print(f"  ✅ Join query returned {len(results)} sample records")
        
        # Test dogs table aggregation
        cursor.execute("""
            SELECT 
                dog_name,
                total_races,
                total_wins,
                total_places
            FROM dogs
            WHERE total_races > 0
            LIMIT 3
        """)
        
        dogs = cursor.fetchall()
        print(f"  ✅ Dogs aggregation table has {len(dogs)} sample records")
        
        conn.close()
        print("✅ Database schema tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_ml_system_import():
    """Test ML System V3 imports and basic initialization"""
    print("\n🔍 Testing ML System V3 import and initialization...")
    
    try:
        from ml_system_v3 import MLSystemV3
        
        # Initialize with test database
        ml_system = MLSystemV3('greyhound_racing_data_test.db')
        print("  ✅ ML System V3 initialized successfully")
        
        # Test data loading
        data = ml_system._load_comprehensive_data()
        print(f"  ✅ Loaded {len(data)} records for ML training")
        
        if len(data) > 100:
            print("  ✅ Sufficient data available for ML training")
        else:
            print("  ⚠️ Limited data available for ML training")
        
        return True
        
    except Exception as e:
        print(f"❌ ML System test failed: {e}")
        return False

def test_app_integration():
    """Test Flask app database manager integration"""
    print("\n🔍 Testing Flask app database integration...")
    
    try:
        # Add current directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import app components
        from app import DatabaseManager
        
        # Initialize database manager with test database
        db_manager = DatabaseManager('greyhound_racing_data_test.db')
        
        # Test basic operations
        stats = db_manager.get_database_stats()
        print(f"  ✅ Database stats: {stats}")
        
        # Test race details
        recent_races = db_manager.get_recent_races(limit=3)
        print(f"  ✅ Retrieved {len(recent_races)} recent races")
        
        return True
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Unified Schema Compatibility Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_database_connection():
        tests_passed += 1
    
    if test_ml_system_import():
        tests_passed += 1
    
    if test_app_integration():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All unified schema tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
