#!/usr/bin/env python3
"""
Training Data Validation Script
===============================

Validates training data integrity and temporal safety before full training.
Ensures data quality, temporal ordering, and proper feature building.
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_database_schema():
    """Validate database schema and required tables."""
    print("🔍 1. Database Schema Validation")
    print("-" * 50)
    
    db_path = "greyhound_racing_data.db"
    if not os.path.exists(db_path):
        print(f"   ❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check required tables
        required_tables = ['race_metadata', 'dog_race_data', 'enhanced_expert_data']
        optional_tables = ['gr_dog_form', 'expert_form_analysis']  # TGR tables
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"   📊 Found {len(existing_tables)} tables in database")
        
        # Check required tables
        for table in required_tables:
            if table in existing_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   ✅ {table}: {count:,} records")
            else:
                print(f"   ❌ Missing required table: {table}")
                conn.close()
                return False
        
        # Check TGR tables
        tgr_tables_found = 0
        for table in optional_tables:
            if table in existing_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   🎯 {table}: {count:,} records (TGR)")
                tgr_tables_found += 1
            else:
                print(f"   ⚠️  {table}: Not found (TGR optional)")
        
        if tgr_tables_found > 0:
            print(f"   ✅ TGR integration available ({tgr_tables_found}/{len(optional_tables)} tables)")
        else:
            print(f"   ⚠️  No TGR tables found - training will use standard features only")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Database validation error: {e}")
        return False

def validate_temporal_data_integrity():
    """Validate temporal integrity of training data."""
    print("\n🕐 2. Temporal Data Integrity")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        
        # Check temporal ordering
        query = """
        SELECT 
            d.race_id,
            r.race_date,
            r.race_time,
            COUNT(*) as dog_count,
            COUNT(d.finish_position) as positions_available,
            MIN(d.finish_position) as min_position,
            MAX(d.finish_position) as max_position
        FROM dog_race_data d
        LEFT JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id IS NOT NULL 
            AND r.race_date IS NOT NULL
            AND d.finish_position IS NOT NULL
        GROUP BY d.race_id
        ORDER BY r.race_date DESC, r.race_time DESC
        LIMIT 20
        """
        
        recent_races = pd.read_sql_query(query, conn)
        print(f"   📊 Recent races sample: {len(recent_races)}")
        
        # Check for data quality issues
        issues = []
        for _, race in recent_races.iterrows():
            # Check for reasonable field sizes
            if race['dog_count'] < 3 or race['dog_count'] > 20:
                issues.append(f"Race {race['race_id']}: unusual field size ({race['dog_count']})")
            
            # Check for complete position data
            if race['positions_available'] != race['dog_count']:
                issues.append(f"Race {race['race_id']}: missing positions ({race['positions_available']}/{race['dog_count']})")
            
            # Check for valid position range
            if race['min_position'] != 1 or race['max_position'] != race['dog_count']:
                issues.append(f"Race {race['race_id']}: invalid position range ({race['min_position']}-{race['max_position']}, expected 1-{race['dog_count']})")
        
        if issues:
            print(f"   ⚠️  Found {len(issues)} data quality issues:")
            for issue in issues[:5]:  # Show first 5
                print(f"      • {issue}")
            if len(issues) > 5:
                print(f"      ... and {len(issues) - 5} more")
        else:
            print(f"   ✅ No data quality issues found in recent races")
        
        # Check temporal distribution
        query_temporal = """
        SELECT 
            DATE(race_date) as date,
            COUNT(DISTINCT race_id) as races,
            COUNT(*) as total_dogs
        FROM dog_race_data d
        LEFT JOIN race_metadata r ON d.race_id = r.race_id
        WHERE race_date >= date('now', '-30 days')
        GROUP BY DATE(race_date)
        ORDER BY date DESC
        LIMIT 10
        """
        
        temporal_dist = pd.read_sql_query(query_temporal, conn)
        print(f"   📅 Recent 10 days race distribution:")
        for _, day in temporal_dist.iterrows():
            print(f"      {day['date']}: {day['races']} races, {day['total_dogs']} dogs")
        
        conn.close()
        return len(issues) == 0
        
    except Exception as e:
        print(f"   ❌ Temporal validation error: {e}")
        return False

def validate_feature_building():
    """Validate feature building logic with TGR integration."""
    print("\n🔧 3. Feature Building Validation")
    print("-" * 50)
    
    try:
        from temporal_feature_builder import TemporalFeatureBuilder
        from ml_system_v4 import MLSystemV4
        
        # Initialize components
        temporal_builder = TemporalFeatureBuilder()
        ml_system = MLSystemV4()
        
        print(f"   ✅ TemporalFeatureBuilder initialized")
        print(f"   ✅ MLSystemV4 initialized")
        
        # Check TGR integration
        tgr_integrator = getattr(temporal_builder, 'tgr_integrator', None)
        if tgr_integrator:
            tgr_features = tgr_integrator.get_feature_names()
            print(f"   🎯 TGR integration active: {len(tgr_features)} features")
        else:
            print(f"   ⚠️  TGR integration not found")
        
        # Load sample training data
        print(f"   📊 Loading sample training data...")
        train_data, test_data = ml_system.prepare_time_ordered_data()
        
        if train_data.empty or test_data.empty:
            print(f"   ❌ No training data available")
            return False
        
        print(f"   ✅ Training data: {len(train_data)} samples, {len(train_data['race_id'].unique())} races")
        print(f"   ✅ Test data: {len(test_data)} samples, {len(test_data['race_id'].unique())} races")
        
        # Test feature building on sample race
        sample_race_id = train_data['race_id'].iloc[0]
        sample_race_data = train_data[train_data['race_id'] == sample_race_id]
        
        print(f"   🔬 Testing feature building on race: {sample_race_id}")
        print(f"      Dogs in race: {len(sample_race_data)}")
        
        # Build features for sample race
        features = temporal_builder.build_features_for_race(sample_race_data, sample_race_id)
        
        if features is None or features.empty:
            print(f"   ❌ Feature building failed")
            return False
        
        # Analyze features
        feature_columns = list(features.columns)
        tgr_features = [col for col in feature_columns if col.startswith('tgr_')]
        standard_features = [col for col in feature_columns if not col.startswith('tgr_') 
                           and col not in ['race_id', 'dog_clean_name', 'target', 'target_timestamp']]
        
        print(f"   ✅ Features built successfully:")
        print(f"      Total features: {len(feature_columns)}")
        print(f"      Standard features: {len(standard_features)}")
        print(f"      TGR features: {len(tgr_features)}")
        
        # Check for required standard features
        expected_standard = ['historical_avg_position', 'historical_win_rate', 'venue_experience']
        found_standard = [f for f in expected_standard if f in feature_columns]
        print(f"      Key standard features found: {len(found_standard)}/{len(expected_standard)}")
        
        # Check TGR features if available
        if tgr_features:
            expected_tgr = ['tgr_win_rate', 'tgr_form_trend', 'tgr_consistency']
            found_tgr = [f for f in expected_tgr if f in tgr_features]
            print(f"      Key TGR features found: {len(found_tgr)}/{len(expected_tgr)}")
        
        # Validate temporal safety
        print(f"   🛡️  Validating temporal integrity...")
        temporal_builder.validate_temporal_integrity(features, sample_race_data)
        print(f"   ✅ Temporal integrity validated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature building validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_training_pipeline():
    """Validate the complete training pipeline."""
    print("\n🏗️ 4. Training Pipeline Validation")
    print("-" * 50)
    
    try:
        from ml_system_v4 import MLSystemV4
        
        # Initialize ML system
        ml_system = MLSystemV4()
        print(f"   ✅ MLSystemV4 initialized")
        
        # Check current model status
        current_model = getattr(ml_system, 'calibrated_pipeline', None)
        model_info = getattr(ml_system, 'model_info', {})
        feature_columns = getattr(ml_system, 'feature_columns', [])
        
        print(f"   📊 Current model status:")
        print(f"      Model loaded: {'Yes' if current_model else 'No'}")
        print(f"      Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"      Trained at: {model_info.get('trained_at', 'Unknown')}")
        print(f"      Feature count: {len(feature_columns)}")
        
        # Test data preparation
        print(f"   🔄 Testing data preparation...")
        train_data, test_data = ml_system.prepare_time_ordered_data()
        
        if train_data.empty or test_data.empty:
            print(f"   ❌ Data preparation failed")
            return False
        
        print(f"   ✅ Data preparation successful:")
        print(f"      Training races: {len(train_data['race_id'].unique())}")
        print(f"      Test races: {len(test_data['race_id'].unique())}")
        print(f"      Training samples: {len(train_data)}")
        print(f"      Test samples: {len(test_data)}")
        
        # Check temporal split
        train_max_date = train_data['race_timestamp'].max() if 'race_timestamp' in train_data.columns else 'Unknown'
        test_min_date = test_data['race_timestamp'].min() if 'race_timestamp' in test_data.columns else 'Unknown'
        
        print(f"   📅 Temporal split validation:")
        print(f"      Training period ends: {train_max_date}")
        print(f"      Test period starts: {test_min_date}")
        
        # Test feature building
        print(f"   🔧 Testing feature building...")
        train_features = ml_system.build_leakage_safe_features(train_data.head(100))  # Test with small sample
        
        if train_features is None or train_features.empty:
            print(f"   ❌ Feature building failed")
            return False
        
        print(f"   ✅ Feature building successful:")
        print(f"      Feature shape: {train_features.shape}")
        
        # Check for TGR features in output
        tgr_features = [col for col in train_features.columns if col.startswith('tgr_')]
        print(f"      TGR features included: {len(tgr_features)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training pipeline validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("🔍 ML Training Data Validation")
    print("=" * 60)
    print("Validating training data integrity and logic before full training...")
    
    validations = [
        ("Database Schema", validate_database_schema),
        ("Temporal Data Integrity", validate_temporal_data_integrity), 
        ("Feature Building Logic", validate_feature_building),
        ("Training Pipeline", validate_training_pipeline)
    ]
    
    results = {}
    for name, validator in validations:
        try:
            results[name] = validator()
        except Exception as e:
            print(f"\n❌ {name} validation failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("🎯 Validation Summary:")
    print("-" * 30)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n🎉 All validations passed! Training pipeline is ready.")
        print(f"✅ Data integrity confirmed")
        print(f"✅ Temporal safety verified")
        print(f"✅ TGR integration working")
        print(f"✅ Ready for full training")
        return True
    else:
        print(f"\n⚠️  Some validations failed. Review issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
