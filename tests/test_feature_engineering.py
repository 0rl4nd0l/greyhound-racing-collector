#!/usr/bin/env python3
"""
Test script for Feature Engineering & Feature Store (Step 5)
============================================================

This script demonstrates the implementation of:
• Versioned feature groups (v3_distance_stats, v3_recent_form, etc.)
• Feature persistence to feature_store.parquet with metadata
• Automatic drift detection using Kolmogorov-Smirnov test
• Integration with the Flask API endpoint

Author: AI Assistant
Date: July 31, 2025
"""

import pandas as pd
import numpy as np
from features import (V3DistanceStatsFeatures, V3RecentFormFeatures, 
                     V3VenueAnalysisFeatures, V3BoxPositionFeatures,
                     V3CompetitionFeatures, V3WeatherTrackFeatures, 
                     V3TrainerFeatures, FeatureStore)
from enhanced_feature_engineering_v2 import EnhancedFeatureEngineer

def test_versioned_features():
    """Test versioned feature group creation"""
    print("🔧 Testing Versioned Feature Groups")
    print("=" * 50)
    
    # Sample dog statistics
    sample_dog_stats = {
        'avg_time': 28.5,
        'best_time': 27.2,
        'races_count': 15,
        'win_rate': 0.25,
        'recent_form': [2, 1, 3, 2, 1],
        'venue_stats': {'SANDOWN': {'races': 8, 'avg_position': 2.1}},
        'box_positions': {1: 3, 2: 2, 3: 4, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1},
        'trainer_stats': {'win_rate': 0.18, 'recent_success': 0.22}
    }
    
    # Test each feature group
    feature_groups = [
        ('Distance Stats', V3DistanceStatsFeatures()),
        ('Recent Form', V3RecentFormFeatures()),
        ('Venue Analysis', V3VenueAnalysisFeatures()),
        ('Box Position', V3BoxPositionFeatures()),
        ('Competition', V3CompetitionFeatures()),
        ('Weather/Track', V3WeatherTrackFeatures()),
        ('Trainer', V3TrainerFeatures())
    ]
    
    all_features = {}
    for name, feature_class in feature_groups:
        features = feature_class.create_features(sample_dog_stats)
        all_features.update(features)
        print(f"✅ {name}: Generated {len(features)} features")
        print(f"   Version: {feature_class.version}")
        print(f"   Sample features: {list(features.keys())[:3]}")
        print()
    
    print(f"📊 Total features generated: {len(all_features)}")
    return all_features

def test_enhanced_feature_engineer():
    """Test enhanced feature engineer using versioned modules"""
    print("🚀 Testing Enhanced Feature Engineer")
    print("=" * 50)
    
    # Sample data
    sample_dog_stats = {
        'avg_time': 28.5,
        'best_time': 27.2,
        'races_count': 15,
        'win_rate': 0.25,
        'recent_form': [2, 1, 3, 2, 1],
        'venue_stats': {'SANDOWN': {'races': 8, 'avg_position': 2.1}},
        'box_positions': {1: 3, 2: 2, 3: 4, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1},
        'trainer_stats': {'win_rate': 0.18, 'recent_success': 0.22}
    }
    
    engineer = EnhancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_dog_stats)
    
    print(f"✅ Enhanced engineer generated {len(features)} features")
    print("Sample features:")
    for i, (key, value) in enumerate(list(features.items())[:5]):
        print(f"  {key}: {value:.4f}")
    
    return features

def test_feature_store():
    """Test feature store persistence and loading"""
    print("💾 Testing Feature Store")
    print("=" * 50)
    
    # Initialize feature store
    fs = FeatureStore()
    
    # Generate sample features
    np.random.seed(999)
    new_features = pd.DataFrame({
        'v3_distance_avg_time': np.random.normal(29.0, 1.5, 50),  # Different from baseline
        'v3_distance_speed_rating': np.random.normal(81, 9, 50),
        'v3_recent_form_trend': np.random.normal(0.08, 0.14, 50),
        'v3_recent_win_rate': np.random.beta(2.5, 5, 50),
        'v3_venue_home_advantage': np.random.normal(0.01, 0.09, 50),
        'v3_box_position_advantage': np.random.normal(0.01, 0.06, 50),
        'v3_competition_strength': np.random.normal(0.52, 0.11, 50),
        'v3_weather_track_impact': np.random.normal(0.005, 0.04, 50),
        'v3_trainer_success_rate': np.random.beta(3.2, 7, 50),
    })
    
    # Persist features
    fs.persist(new_features)
    print(f"✅ Persisted {len(new_features)} feature samples")
    
    # Load features back
    loaded_features = fs.load()
    print(f"✅ Loaded {len(loaded_features)} feature samples")
    
    return new_features

def test_drift_detection():
    """Test drift detection between current and baseline features"""
    print("🔍 Testing Drift Detection")
    print("=" * 50)
    
    fs = FeatureStore()
    
    # Load baseline features
    try:
        baseline_features = pd.read_parquet('baseline_feature_store.parquet')
        print(f"✅ Loaded baseline features: {baseline_features.shape}")
    except FileNotFoundError:
        print("❌ Baseline feature store not found")
        return
    
    # Load current features
    try:
        current_features = fs.load()
        print(f"✅ Loaded current features: {current_features.shape}")
    except FileNotFoundError:
        print("❌ Current feature store not found")
        return
    
    # Check for drift
    drift_results = fs.check_drift(current_features, baseline_features)
    
    print("\n🔍 Drift Detection Results:")
    drift_count = 0
    for feature, result in drift_results.items():
        if result['drift_detected']:
            drift_count += 1
            print(f"🚨 {feature}: DRIFT DETECTED (p={result['p_value']:.4f})")
        else:
            print(f"✅ {feature}: No drift (p={result['p_value']:.4f})")
    
    print(f"\n📊 Summary: {drift_count}/{len(drift_results)} features show drift")
    
    return drift_results

def test_integration():
    """Test integration with existing enhanced feature engineering"""
    print("🔗 Testing Integration")
    print("=" * 50)
    
    # Sample race data
    sample_dogs = [
        {
            'dog_name': 'Test Dog 1',
            'avg_time': 28.5,
            'races_count': 15,
            'win_rate': 0.25,
            'recent_form': [2, 1, 3, 2, 1],
            'venue_stats': {'SANDOWN': {'races': 8, 'avg_position': 2.1}}
        },
        {
            'dog_name': 'Test Dog 2', 
            'avg_time': 29.1,
            'races_count': 12,
            'win_rate': 0.18,
            'recent_form': [3, 4, 2, 3, 2],
            'venue_stats': {'SANDOWN': {'races': 5, 'avg_position': 3.2}}
        }
    ]
    
    engineer = EnhancedFeatureEngineer()
    all_dog_features = []
    
    for dog_stats in sample_dogs:
        features = engineer.create_advanced_features(dog_stats)
        all_dog_features.append(features)
        print(f"✅ Generated {len(features)} features for {dog_stats['dog_name']}")
    
    # Create DataFrame and persist
    feature_df = pd.DataFrame(all_dog_features)
    fs = FeatureStore(path='integration_test_features.parquet')
    fs.persist(feature_df)
    
    print(f"✅ Persisted features for {len(sample_dogs)} dogs")
    print(f"   Feature dimensions: {feature_df.shape}")
    
    return feature_df

def main():
    """Run all tests"""
    print("🧪 Feature Engineering & Feature Store Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Versioned feature groups
        versioned_features = test_versioned_features()
        print()
        
        # Test 2: Enhanced feature engineer
        enhanced_features = test_enhanced_feature_engineer()
        print()
        
        # Test 3: Feature store
        stored_features = test_feature_store()
        print()
        
        # Test 4: Drift detection
        drift_results = test_drift_detection()
        print()
        
        # Test 5: Integration
        integration_features = test_integration()
        print()
        
        print("🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print(f"  • Versioned features: {len(versioned_features)} features")
        print(f"  • Enhanced engineer: {len(enhanced_features)} features")
        print(f"  • Feature store: {len(stored_features)} samples persisted")
        print(f"  • Drift detection: {sum(1 for r in drift_results.values() if r['drift_detected'])}/{len(drift_results)} features with drift")
        print(f"  • Integration test: {integration_features.shape[0]} dogs, {integration_features.shape[1]} features")
        
        print("\n✅ Step 5: Feature Engineering & Feature Store - COMPLETED")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
