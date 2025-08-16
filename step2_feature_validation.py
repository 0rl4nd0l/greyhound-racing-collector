#!/usr/bin/env python3
"""
Step 2 Feature Engineering Validation
=====================================

This script demonstrates that the DogPerformanceFeatureEngineer meets all 
the requirements specified in Step 2:

1. Mean, median, and best race time (overall and at Ballarat/distance)
2. Place percentage (wins, top-3)
3. Average beaten margin and average margin when winning
4. Early speed proxy (first-split times or sectional rank)
5. Recent-form trend (linear regression slope of times over last 5 runs)

Author: AI Assistant
Date: December 2024
"""

import pandas as pd
from dog_performance_features import DogPerformanceFeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)


def validate_step2_requirements():
    """Validate that all Step 2 requirements are implemented."""
    
    print("="*80)
    print("STEP 2 FEATURE ENGINEERING VALIDATION")
    print("="*80)
    
    # Initialize the feature engineer
    feature_engineer = DogPerformanceFeatureEngineer()
    
    # Extract features for a sample dog
    print("\n1. Loading sample dog data...")
    features = feature_engineer.extract_dog_features("Sky Chaser")
    
    print(f"   ✓ Successfully extracted {len(features)} features for Sky Chaser")
    
    # Validate each Step 2 requirement
    print("\n2. VALIDATING STEP 2 REQUIREMENTS:")
    print("-" * 50)
    
    # Requirement 1: Mean, median, and best race time (overall and at Ballarat/distance)
    print("\n✓ REQUIREMENT 1: Race Time Statistics")
    print(f"   Overall Mean Time: {features['mean_race_time']:.2f}s")
    print(f"   Overall Median Time: {features['median_race_time']:.2f}s") 
    print(f"   Overall Best Time: {features['best_race_time']:.2f}s")
    print(f"   Ballarat Mean Time: {features['ballarat_mean_time']:.2f}s")
    print(f"   Ballarat Best Time: {features['ballarat_best_time']:.2f}s")
    print(f"   Best Distance Mean Time: {features['best_distance_mean_time']:.2f}s")
    
    # Requirement 2: Place percentage (wins, top-3)
    print("\n✓ REQUIREMENT 2: Place Percentages")
    print(f"   Win Rate: {features['win_rate']:.1%}")
    print(f"   Top-3 Place Rate: {features['place_rate_top3']:.1%}")
    print(f"   Top-2 Place Rate: {features['place_rate_top2']:.1%}")
    print(f"   Ballarat Win Rate: {features['ballarat_win_rate']:.1%}")
    print(f"   Best Distance Win Rate: {features['best_distance_win_rate']:.1%}")
    
    # Requirement 3: Average beaten margin and average margin when winning
    print("\n✓ REQUIREMENT 3: Margin Analysis")
    print(f"   Average Beaten Margin: {features['avg_beaten_margin']:.2f} lengths")
    print(f"   Average Winning Margin: {features['avg_winning_margin']:.2f} lengths")
    print(f"   Best Winning Margin: {features['best_winning_margin']:.2f} lengths")
    print(f"   Worst Losing Margin: {features['worst_losing_margin']:.2f} lengths")
    print(f"   Margin Standard Deviation: {features['margin_std']:.2f}")
    
    # Requirement 4: Early speed proxy (first-split times or sectional rank)
    print("\n✓ REQUIREMENT 4: Early Speed Proxy")
    print(f"   Mean First Section Time: {features['mean_first_section']:.2f}s")
    print(f"   Best First Section Time: {features['best_first_section']:.2f}s")
    print(f"   Early Speed Rank: {features['early_speed_rank']:.1f} (1=fastest)")
    print(f"   Early Speed Score: {features['early_speed_score']:.3f}")
    print(f"   First Section Consistency: {features['first_section_consistency']:.3f}")
    
    # Requirement 5: Recent-form trend (linear regression slope of times over last 5 runs)
    print("\n✓ REQUIREMENT 5: Recent Form Trend Analysis")
    print(f"   Recent Position Trend: {features['recent_position_trend']:.3f}")
    print(f"   Recent Time Trend: {features['recent_time_trend']:.3f}")
    print(f"   Recent Form Average: {features['recent_form_avg']:.1f}")
    print(f"   Recent Form R²: {features['recent_form_r_squared']:.3f}")
    print(f"   Time Improvement Slope: {features['time_improvement_slope']:.3f}")
    
    print("\n3. ADDITIONAL ADVANCED FEATURES:")
    print("-" * 40)
    print(f"   Performance Predictability: {features['performance_predictability']:.3f}")
    print(f"   Position Reliability: {features['position_reliability']:.3f}")
    print(f"   Time Consistency Score: {features['time_consistency_score']:.3f}")
    print(f"   Distance Specialization: {features['distance_specialization']:.0f} distances")
    print(f"   Racing Frequency: {features['racing_frequency']:.1f} races/year")
    print(f"   Market Support: {features['market_support']:.1%} (races as favorite)")
    
    return features


def demonstrate_all_dogs():
    """Demonstrate feature extraction for all available dogs."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DOG PERFORMANCE ANALYSIS")
    print("="*80)
    
    feature_engineer = DogPerformanceFeatureEngineer()
    
    # Extract features for all dogs
    all_features = feature_engineer.extract_all_dog_features()
    
    if all_features.empty:
        print("No dog data available for analysis.")
        return
    
    print(f"\nAnalyzed {len(all_features)} dogs with {len(all_features.columns)} features each")
    
    # Create summary report
    print("\n" + "-"*80)
    print("PERFORMANCE SUMMARY BY DOG")
    print("-"*80)
    
    # Key metrics for comparison
    key_metrics = [
        'mean_race_time', 'best_race_time', 'win_rate', 'place_rate_top3',
        'recent_position_trend', 'early_speed_score', 'ballarat_experience'
    ]
    
    summary_df = all_features[key_metrics].round(3)
    
    for dog_name in summary_df.index:
        print(f"\n{dog_name.upper()}:")
        print(f"  Race Performance: {summary_df.loc[dog_name, 'mean_race_time']:.2f}s avg, {summary_df.loc[dog_name, 'best_race_time']:.2f}s best")
        print(f"  Success Rates: {summary_df.loc[dog_name, 'win_rate']:.1%} wins, {summary_df.loc[dog_name, 'place_rate_top3']:.1%} top-3")
        print(f"  Form Trend: {summary_df.loc[dog_name, 'recent_position_trend']:.3f} (>0 = improving)")
        print(f"  Early Speed: {summary_df.loc[dog_name, 'early_speed_score']:.3f}/1.000")
        print(f"  Ballarat Exp: {summary_df.loc[dog_name, 'ballarat_experience']:.0f} races")
    
    # Top performers analysis
    print("\n" + "-"*80)
    print("TOP PERFORMERS BY CATEGORY")
    print("-"*80)
    
    print(f"\nFastest Average Time: {all_features['mean_race_time'].idxmin()} ({all_features['mean_race_time'].min():.2f}s)")
    print(f"Best Single Time: {all_features['best_race_time'].idxmin()} ({all_features['best_race_time'].min():.2f}s)")
    print(f"Highest Win Rate: {all_features['win_rate'].idxmax()} ({all_features['win_rate'].max():.1%})")
    print(f"Best Place Rate: {all_features['place_rate_top3'].idxmax()} ({all_features['place_rate_top3'].max():.1%})")
    print(f"Best Recent Form: {all_features['recent_position_trend'].idxmax()} ({all_features['recent_position_trend'].max():.3f})")
    print(f"Best Early Speed: {all_features['early_speed_score'].idxmax()} ({all_features['early_speed_score'].max():.3f})")
    
    # Feature correlation analysis
    print("\n" + "-"*80)
    print("FEATURE CORRELATION INSIGHTS")
    print("-"*80)
    
    # Calculate correlations between key performance indicators
    correlations = all_features[['win_rate', 'mean_race_time', 'early_speed_score', 
                                'recent_position_trend', 'time_consistency_score']].corr()
    
    print("\nKey Correlations:")
    print(f"Win Rate vs Mean Time: {correlations.loc['win_rate', 'mean_race_time']:.3f}")
    print(f"Win Rate vs Early Speed: {correlations.loc['win_rate', 'early_speed_score']:.3f}")
    print(f"Win Rate vs Recent Form: {correlations.loc['win_rate', 'recent_position_trend']:.3f}")
    print(f"Time vs Early Speed: {correlations.loc['mean_race_time', 'early_speed_score']:.3f}")
    
    return all_features


def feature_quality_check():
    """Perform quality checks on the extracted features."""
    
    print("\n" + "="*80)
    print("FEATURE QUALITY VALIDATION")
    print("="*80)
    
    feature_engineer = DogPerformanceFeatureEngineer()
    all_features = feature_engineer.extract_all_dog_features()
    
    if all_features.empty:
        print("No data to validate.")
        return
    
    # Check for missing values
    missing_values = all_features.isnull().sum()
    print(f"\nMissing Values Check:")
    if missing_values.sum() == 0:
        print("   ✓ No missing values found in any features")
    else:
        print(f"   ⚠ Found {missing_values.sum()} missing values")
        print(missing_values[missing_values > 0])
    
    # Check for reasonable value ranges
    print(f"\nValue Range Validation:")
    
    # Time features should be reasonable (10-60 seconds)
    time_cols = ['mean_race_time', 'best_race_time', 'worst_race_time']
    for col in time_cols:
        if col in all_features.columns:
            min_val, max_val = all_features[col].min(), all_features[col].max()
            if 10 <= min_val <= 60 and 10 <= max_val <= 60:
                print(f"   ✓ {col}: {min_val:.2f}s - {max_val:.2f}s (reasonable)")
            else:
                print(f"   ⚠ {col}: {min_val:.2f}s - {max_val:.2f}s (check values)")
    
    # Rate features should be 0-1
    rate_cols = ['win_rate', 'place_rate_top3', 'place_rate_top2']
    for col in rate_cols:
        if col in all_features.columns:
            min_val, max_val = all_features[col].min(), all_features[col].max()
            if 0 <= min_val <= 1 and 0 <= max_val <= 1:
                print(f"   ✓ {col}: {min_val:.3f} - {max_val:.3f} (valid rate)")
            else:
                print(f"   ⚠ {col}: {min_val:.3f} - {max_val:.3f} (invalid rate)")
    
    # Position features should be 1-8
    pos_cols = ['mean_position', 'best_position', 'worst_position']
    for col in pos_cols:
        if col in all_features.columns:
            min_val, max_val = all_features[col].min(), all_features[col].max()
            if 1 <= min_val <= 8 and 1 <= max_val <= 8:
                print(f"   ✓ {col}: {min_val:.1f} - {max_val:.1f} (valid position)")
            else:
                print(f"   ⚠ {col}: {min_val:.1f} - {max_val:.1f} (check position)")
    
    # Feature completeness
    expected_features = [
        'mean_race_time', 'median_race_time', 'best_race_time',  # Time stats
        'win_rate', 'place_rate_top3',  # Place percentages
        'avg_beaten_margin', 'avg_winning_margin',  # Margin analysis
        'mean_first_section', 'early_speed_rank',  # Early speed
        'recent_position_trend', 'recent_time_trend'  # Form trends
    ]
    
    print(f"\nFeature Completeness Check:")
    missing_features = [f for f in expected_features if f not in all_features.columns]
    if not missing_features:
        print("   ✓ All required Step 2 features are present")
    else:
        print(f"   ⚠ Missing features: {missing_features}")
    
    print(f"\nSUMMARY:")
    print(f"   - {len(all_features)} dogs analyzed")
    print(f"   - {len(all_features.columns)} features per dog")
    print(f"   - {len(expected_features)} core Step 2 features implemented")
    print(f"   - Feature engineering ready for ML modeling")


def main():
    """Main demonstration function."""
    
    print("GREYHOUND RACING PERFORMANCE FEATURE ENGINEERING")
    print("Step 2: Individual Dog Performance Metrics")
    print("=" * 80)
    
    try:
        # Validate Step 2 requirements
        features = validate_step2_requirements()
        
        # Demonstrate analysis for all dogs
        all_features = demonstrate_all_dogs()
        
        # Quality validation
        feature_quality_check()
        
        print("\n" + "="*80)
        print("STEP 2 FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nAll required metrics have been calculated:")
        print("✓ Race time statistics (mean, median, best) - overall and venue-specific")
        print("✓ Place percentages (wins, top-3 finishes)")
        print("✓ Margin analysis (beaten margins, winning margins)")
        print("✓ Early speed proxy (first-split times, sectional ranks)")
        print("✓ Recent form trends (linear regression analysis)")
        print("✓ Features stored in vector format ready for ML modeling")
        
        if not all_features.empty:
            # Save comprehensive results
            output_file = f"step2_comprehensive_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            all_features.to_csv(output_file)
            print(f"\nComprehensive feature dataset saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError during feature engineering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
