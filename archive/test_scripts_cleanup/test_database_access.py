#!/usr/bin/env python3
"""
Database Access Test
===================

Test database connectivity and data access by the pipeline components.
"""

import sqlite3
import pandas as pd
from enhanced_feature_engineering_v2 import AdvancedFeatureEngineer

def test_database_connectivity():
    """Test basic database connectivity"""
    print("🔍 Testing Database Connectivity...")
    
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        
        # Test basic queries
        cursor = conn.cursor()
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]
        print(f"✅ Race metadata records: {race_count}")
        
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        dog_count = cursor.fetchone()[0]
        print(f"✅ Dog race data records: {dog_count}")
        
        # Test join query
        cursor.execute("""
            SELECT COUNT(*) FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.individual_time IS NOT NULL
        """)
        valid_records = cursor.fetchone()[0]
        print(f"✅ Valid joined records: {valid_records}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connectivity failed: {e}")
        return False

def test_feature_engineer_data_access():
    """Test feature engineer's data access"""
    print("\n🔍 Testing Feature Engineer Data Access...")
    
    try:
        engineer = AdvancedFeatureEngineer("greyhound_racing_data.db")
        
        # Load comprehensive data
        df = engineer.load_comprehensive_data()
        
        print(f"✅ Loaded {len(df)} records")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        print(f"✅ Unique dogs: {df['dog_clean_name'].nunique()}")
        print(f"✅ Unique venues: {df['venue'].nunique()}")
        
        # Test sample feature creation
        if len(df) > 0:
            sample_dog = df['dog_clean_name'].iloc[0]
            sample_date = df['race_date'].iloc[0]
            sample_venue = df['venue'].iloc[0]
            
            print(f"\n🧪 Testing feature creation for: {sample_dog}")
            features = engineer.create_advanced_dog_features(
                df, sample_dog, sample_date, sample_venue
            )
            
            print(f"✅ Generated {len(features)} features")
            for key, value in list(features.items())[:5]:
                print(f"   {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Feature engineer test failed: {e}")
        import traceback
        print(f"   {traceback.format_exc()}")
        return False

def test_data_quality():
    """Test data quality issues"""
    print("\n🔍 Testing Data Quality...")
    
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        
        # Check for common data issues
        issues = {}
        
        # Missing finish positions
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NULL OR finish_position = ''")
        issues['missing_positions'] = cursor.fetchone()[0]
        
        # Missing times
        cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE individual_time IS NULL OR individual_time = ''")
        issues['missing_times'] = cursor.fetchone()[0]
        
        # Invalid box numbers
        cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE box_number IS NULL OR box_number <= 0")
        issues['invalid_boxes'] = cursor.fetchone()[0]
        
        # Missing venues
        cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE venue IS NULL OR venue = ''")
        issues['missing_venues'] = cursor.fetchone()[0]
        
        print("📊 Data Quality Issues:")
        for issue, count in issues.items():
            status = "⚠️" if count > 0 else "✅"
            print(f"   {status} {issue}: {count}")
        
        # Sample of recent valid data
        cursor.execute("""
            SELECT drd.dog_clean_name, drd.finish_position, drd.individual_time, 
                   rm.venue, rm.race_date
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
                AND drd.individual_time IS NOT NULL
                AND drd.box_number IS NOT NULL
                AND rm.venue IS NOT NULL
            ORDER BY rm.race_date DESC
            LIMIT 5
        """)
        
        print("\n📝 Sample Valid Records:")
        for row in cursor.fetchall():
            print(f"   {row[0]} - Pos: {row[1]}, Time: {row[2]}, Venue: {row[3]}, Date: {row[4]}")
        
        conn.close()
        return sum(issues.values()) == 0
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False

def test_pipeline_database_usage():
    """Test how the pipeline uses the database"""
    print("\n🔍 Testing Pipeline Database Usage...")
    
    try:
        from enhanced_pipeline_v2 import EnhancedPipelineV2
        
        pipeline = EnhancedPipelineV2("greyhound_racing_data.db")
        
        # Check if feature engineer is available and working
        if pipeline.feature_engineer:
            print("✅ Feature engineer initialized")
            
            # Test data loading
            try:
                data = pipeline.feature_engineer.load_comprehensive_data()
                print(f"✅ Pipeline can access {len(data)} database records")
                
                # Test feature creation with database data
                if len(data) > 0:
                    sample_dog = data['dog_clean_name'].iloc[0]
                    features = pipeline.feature_engineer.create_advanced_dog_features(
                        data, sample_dog, "2025-07-01", "GEE"
                    )
                    print(f"✅ Pipeline generated {len(features)} features from database")
                
            except Exception as e:
                print(f"⚠️ Pipeline database access issue: {e}")
        
        else:
            print("❌ Feature engineer not available in pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline database test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Database Access Test Suite")
    print("=" * 50)
    
    tests = [
        ("Database Connectivity", test_database_connectivity),
        ("Feature Engineer Data Access", test_feature_engineer_data_access),
        ("Data Quality", test_data_quality),
        ("Pipeline Database Usage", test_pipeline_database_usage)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    total_passed = sum(results.values())
    print(f"\n🎯 Overall: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("🎉 All database tests passed!")
    else:
        print("⚠️ Some database issues detected - check logs above")
