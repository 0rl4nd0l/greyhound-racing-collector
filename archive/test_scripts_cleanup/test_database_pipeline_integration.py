#!/usr/bin/env python3
"""
Comprehensive Database & Pipeline Integration Test
=================================================

Test the complete integration between database, feature engineering, and pipeline.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def test_database_integrity():
    """Test database integrity after fixes"""
    print("🔍 Testing Database Integrity After Fixes...")
    
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()
        
        # Count clean records
        cursor.execute("""
            SELECT COUNT(*) FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
                AND drd.finish_position > 0
                AND drd.individual_time IS NOT NULL
                AND drd.individual_time > 0
                AND drd.box_number IS NOT NULL
                AND drd.box_number > 0
                AND rm.venue IS NOT NULL
        """)
        clean_records = cursor.fetchone()[0]
        print(f"✅ Clean, valid records: {clean_records}")
        
        # Check data ranges
        cursor.execute("SELECT MIN(finish_position), MAX(finish_position) FROM dog_race_data WHERE finish_position IS NOT NULL")
        min_pos, max_pos = cursor.fetchone()
        print(f"✅ Finish position range: {min_pos} to {max_pos}")
        
        cursor.execute("SELECT MIN(individual_time), MAX(individual_time) FROM dog_race_data WHERE individual_time IS NOT NULL")
        min_time, max_time = cursor.fetchone()
        print(f"✅ Time range: {float(min_time):.3f}s to {float(max_time):.3f}s")
        
        cursor.execute("SELECT MIN(box_number), MAX(box_number) FROM dog_race_data WHERE box_number IS NOT NULL")
        min_box, max_box = cursor.fetchone()
        print(f"✅ Box number range: {min_box} to {max_box}")
        
        # Sample recent data
        cursor.execute("""
            SELECT drd.dog_clean_name, drd.finish_position, drd.individual_time, 
                   drd.box_number, rm.venue, rm.race_date
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            ORDER BY rm.race_date DESC
            LIMIT 3
        """)
        
        print(f"\n📝 Sample Clean Records:")
        for row in cursor.fetchall():
            print(f"   {row[0]} - Pos:{row[1]}, Time:{row[2]:.3f}s, Box:{row[3]}, {row[4]}, {row[5]}")
        
        conn.close()
        return clean_records > 5000  # Reasonable threshold
        
    except Exception as e:
        print(f"❌ Database integrity test failed: {e}")
        return False

def test_feature_engineering_with_clean_data():
    """Test feature engineering with cleaned data"""
    print("\n🔍 Testing Feature Engineering with Clean Data...")
    
    try:
        from enhanced_feature_engineering_v2 import AdvancedFeatureEngineer
        
        engineer = AdvancedFeatureEngineer("greyhound_racing_data.db")
        df = engineer.load_comprehensive_data()
        
        print(f"✅ Loaded {len(df)} clean records")
        
        # Test feature generation with real data
        if len(df) > 0:
            # Pick a dog with some history
            dog_counts = df['dog_clean_name'].value_counts()
            test_dog = dog_counts.index[0] if len(dog_counts) > 0 else df['dog_clean_name'].iloc[0]
            test_venue = df['venue'].iloc[0]
            test_date = df['race_date'].max()
            
            print(f"🧪 Testing features for: {test_dog} (has {dog_counts.get(test_dog, 0)} races)")
            
            features = engineer.create_advanced_dog_features(
                df, test_dog, test_date, test_venue
            )
            
            print(f"✅ Generated {len(features)} features:")
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # Validate feature quality
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            if len(numeric_features) >= 10:
                print(f"✅ Sufficient numeric features: {len(numeric_features)}")
                
                # Check for reasonable values
                reasonable_count = sum(1 for v in numeric_features.values() 
                                     if not np.isnan(v) and not np.isinf(v))
                print(f"✅ Reasonable values: {reasonable_count}/{len(numeric_features)}")
                
                return reasonable_count >= len(numeric_features) * 0.8
            
        return False
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        import traceback
        print(f"   {traceback.format_exc()}")
        return False

def test_pipeline_with_database():
    """Test complete pipeline with database integration"""
    print("\n🔍 Testing Complete Pipeline with Database...")
    
    try:
        from enhanced_pipeline_v2 import EnhancedPipelineV2
        
        # Initialize pipeline
        pipeline = EnhancedPipelineV2("greyhound_racing_data.db")
        
        # Find a sample race file
        sample_files = list(Path('.').glob('processed/completed/Race*.csv'))
        if not sample_files:
            sample_files = list(Path('.').glob('*Race*.csv'))
        
        if not sample_files:
            print("⚠️ No race files found for testing")
            return False
            
        sample_file = str(sample_files[0])
        print(f"📄 Testing with: {sample_file}")
        
        # Run prediction
        result = pipeline.predict_race_file(sample_file)
        
        if result.get('success'):
            predictions = result.get('predictions', [])
            print(f"✅ Pipeline generated {len(predictions)} predictions")
            
            if predictions:
                # Check prediction quality
                scores = [p['prediction_score'] for p in predictions]
                unique_scores = len(set([round(s, 3) for s in scores]))
                
                print(f"✅ Score range: {min(scores):.3f} to {max(scores):.3f}")
                print(f"✅ Unique scores: {unique_scores}/{len(scores)}")
                
                # Check if database features were used
                enhanced_used = any(p.get('enhanced_features_used', False) for p in predictions)
                print(f"✅ Enhanced features used: {enhanced_used}")
                
                # Show top predictions
                print(f"\n🏆 Top 3 Predictions:")
                for i, pred in enumerate(predictions[:3], 1):
                    print(f"   {i}. {pred['dog_name']}: {pred['prediction_score']:.3f} ({pred['confidence_level']})")
                
                return unique_scores >= 2 and enhanced_used  # Good differentiation and database usage
            
        else:
            print(f"❌ Pipeline failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        print(f"   {traceback.format_exc()}")
        return False

def test_ml_model_with_database_features():
    """Test ML model performance with database-derived features"""
    print("\n🔍 Testing ML Model with Database Features...")
    
    try:
        from enhanced_feature_engineering_v2 import AdvancedFeatureEngineer
        from advanced_ml_system_v2 import AdvancedMLSystemV2
        
        # Get database features
        engineer = AdvancedFeatureEngineer("greyhound_racing_data.db")
        df = engineer.load_comprehensive_data()
        
        if len(df) < 10:
            print("⚠️ Insufficient data for ML testing")
            return False
        
        # Pick test dogs
        test_dogs = df['dog_clean_name'].value_counts().head(3).index.tolist()
        
        # Generate features for test dogs
        test_features = []
        for dog in test_dogs:
            dog_data = df[df['dog_clean_name'] == dog]
            if len(dog_data) > 0:
                venue = dog_data['venue'].iloc[0]
                date = dog_data['race_date'].max()
                
                features = engineer.create_advanced_dog_features(df, dog, date, venue)
                test_features.append(features)
        
        if len(test_features) >= 2:
            # Test ML model
            ml_system = AdvancedMLSystemV2()
            
            if ml_system.models:
                predictions = []
                for features in test_features:
                    pred = ml_system.predict_with_ensemble(features)
                    predictions.append(pred)
                
                print(f"✅ ML predictions: {[f'{p:.4f}' for p in predictions]}")
                
                # Check for variation
                score_variation = max(predictions) - min(predictions)
                print(f"✅ Score variation: {score_variation:.4f}")
                
                return score_variation > 0.001  # Some meaningful variation
            else:
                print("⚠️ No ML models loaded")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ ML model test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Comprehensive Database & Pipeline Integration Test")
    print("=" * 60)
    
    tests = [
        ("Database Integrity", test_database_integrity),
        ("Feature Engineering with Clean Data", test_feature_engineering_with_clean_data),
        ("Pipeline with Database", test_pipeline_with_database),
        ("ML Model with Database Features", test_ml_model_with_database_features)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 Integration Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    total_passed = sum(results.values())
    print(f"\n🎯 Overall: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("🎉 Complete database and pipeline integration successful!")
    else:
        print("⚠️ Some integration issues remain - check details above")
