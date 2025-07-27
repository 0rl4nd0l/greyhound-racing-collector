#!/usr/bin/env python3
"""
Robust Venue Performance Analysis
=================================

This system implements proper venue performance analysis with intelligent fallback logic:
1. Primary: Per-dog venue performance comparison (when sufficient cross-venue data exists)
2. Secondary: Venue difficulty based on field characteristics 
3. Tertiary: Default neutral venue effects with venue-specific metadata

Handles limited data scenarios gracefully while being ready for expanded datasets.

Author: AI Assistant
Date: July 24, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RobustVenueAnalyzer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.venue_characteristics = {}
        self.venue_effects = {}
        self.analysis_confidence = {}
        
    def load_comprehensive_data(self):
        """Load all available race data with metadata"""
        conn = sqlite3.connect(self.db_path)
        
        # Get comprehensive race data
        query = """
        SELECT 
            d.dog_name,
            d.finish_position,
            d.box_number,
            r.venue,
            r.field_size,
            r.race_date,
            r.distance,
            r.grade,
            r.track_condition,
            d.race_id
        FROM dog_race_data d
        JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.finish_position IS NOT NULL 
        AND d.dog_name IS NOT NULL 
        AND d.dog_name != '' 
        AND d.dog_name != 'nan'
        AND r.venue IS NOT NULL
        ORDER BY d.dog_name, r.race_date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"📊 Loaded {len(df)} race entries from {df['race_date'].nunique()} unique dates")
        print(f"📅 Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        
        return df
    
    def analyze_data_coverage(self, df):
        """Analyze data coverage to determine analysis approach"""
        
        # Cross-venue racing analysis
        dog_venues = df.groupby('dog_name')['venue'].nunique()
        multi_venue_dogs = (dog_venues > 1).sum()
        total_dogs = len(dog_venues)
        
        # Venue data analysis
        venue_stats = df.groupby('venue').agg({
            'dog_name': 'nunique',
            'race_id': 'nunique', 
            'field_size': ['mean', 'std'],
            'finish_position': 'count'
        }).round(2)
        
        coverage_stats = {
            'total_dogs': total_dogs,
            'multi_venue_dogs': multi_venue_dogs,
            'cross_venue_percentage': (multi_venue_dogs / total_dogs * 100) if total_dogs > 0 else 0,
            'total_venues': df['venue'].nunique(),
            'date_span_days': (pd.to_datetime(df['race_date'].max()) - pd.to_datetime(df['race_date'].min())).days + 1
        }
        
        print(f"\n📊 DATA COVERAGE ANALYSIS:")
        print(f"   Total dogs: {coverage_stats['total_dogs']}")
        print(f"   Dogs racing at multiple venues: {coverage_stats['multi_venue_dogs']} ({coverage_stats['cross_venue_percentage']:.1f}%)")
        print(f"   Total venues: {coverage_stats['total_venues']}")
        print(f"   Date span: {coverage_stats['date_span_days']} days")
        
        return coverage_stats, venue_stats
    
    def calculate_primary_venue_effects(self, df):
        """Calculate per-dog venue effects (when sufficient cross-venue data)"""
        
        print("\n🎯 CALCULATING PRIMARY VENUE EFFECTS (Per-Dog Analysis)")
        
        # Only analyze dogs with races at multiple venues
        dog_venue_counts = df.groupby('dog_name')['venue'].nunique()
        multi_venue_dogs = dog_venue_counts[dog_venue_counts > 1].index
        
        if len(multi_venue_dogs) == 0:
            print("   ⚠️  No dogs race at multiple venues - skipping primary analysis")
            return pd.DataFrame(), {}
        
        multi_venue_data = df[df['dog_name'].isin(multi_venue_dogs)].copy()
        
        # Calculate each dog's overall performance
        dog_overall = multi_venue_data.groupby('dog_name').agg({
            'finish_position': 'mean',
            'field_size': 'mean'
        }).reset_index()
        dog_overall.columns = ['dog_name', 'overall_avg_position', 'overall_avg_field_size']
        dog_overall['overall_performance_score'] = (
            dog_overall['overall_avg_field_size'] - dog_overall['overall_avg_position'] + 1
        ) / dog_overall['overall_avg_field_size']
        
        # Calculate venue-specific performance
        dog_venue_performance = multi_venue_data.groupby(['dog_name', 'venue']).agg({
            'finish_position': ['mean', 'count'],
            'field_size': 'mean'
        }).reset_index()
        dog_venue_performance.columns = ['dog_name', 'venue', 'venue_avg_position', 'venue_races', 'venue_avg_field_size']
        dog_venue_performance['venue_performance_score'] = (
            dog_venue_performance['venue_avg_field_size'] - dog_venue_performance['venue_avg_position'] + 1
        ) / dog_venue_performance['venue_avg_field_size']
        
        # Merge and calculate venue effects
        comparison = dog_venue_performance.merge(dog_overall, on='dog_name')
        comparison['venue_effect'] = comparison['venue_performance_score'] - comparison['overall_performance_score']
        
        # Aggregate venue effects (only for dogs with 2+ races at venue)
        reliable_comparisons = comparison[comparison['venue_races'] >= 2]
        
        if len(reliable_comparisons) > 0:
            venue_effects = reliable_comparisons.groupby('venue').agg({
                'venue_effect': ['mean', 'std', 'count'],
                'dog_name': 'nunique'
            }).round(4)
            venue_effects.columns = ['avg_effect', 'effect_std', 'comparisons', 'dogs_analyzed']
            venue_effects['confidence'] = np.minimum(venue_effects['dogs_analyzed'] / 10, 1.0)
            
            print(f"   ✅ Calculated venue effects for {len(venue_effects)} venues")
            return venue_effects, comparison
        else:
            print("   ⚠️  Insufficient data for reliable venue effects")
            return pd.DataFrame(), comparison
    
    def calculate_secondary_venue_characteristics(self, df):
        """Calculate venue difficulty based on field characteristics"""
        
        print("\n🏟️ CALCULATING SECONDARY VENUE CHARACTERISTICS (Field Analysis)")
        
        venue_chars = df.groupby('venue').agg({
            'field_size': ['mean', 'std', 'count'],
            'finish_position': 'count',
            'dog_name': 'nunique',
            'race_id': 'nunique',
            'distance': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'grade': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        }).round(2)
        
        venue_chars.columns = [
            'avg_field_size', 'field_size_std', 'field_size_count',
            'total_entries', 'unique_dogs', 'total_races', 'typical_distance', 'typical_grade'
        ]
        
        # Calculate venue difficulty metrics
        venue_chars['dogs_per_race'] = venue_chars['unique_dogs'] / venue_chars['total_races']
        venue_chars['competitiveness'] = venue_chars['dogs_per_race'] / venue_chars['avg_field_size']
        
        # Normalize field size impact (larger fields = harder to win)
        max_field = venue_chars['avg_field_size'].max()
        min_field = venue_chars['avg_field_size'].min()
        
        if max_field > min_field:
            venue_chars['field_difficulty'] = (venue_chars['avg_field_size'] - min_field) / (max_field - min_field)
        else:
            venue_chars['field_difficulty'] = 0.5
        
        # Calculate confidence based on sample size
        venue_chars['data_confidence'] = np.minimum(venue_chars['total_races'] / 20, 1.0)
        
        print(f"   ✅ Calculated characteristics for {len(venue_chars)} venues")
        return venue_chars
    
    def create_venue_performance_system(self, df):
        """Create comprehensive venue performance system with fallbacks"""
        
        print("\n🎯 CREATING COMPREHENSIVE VENUE PERFORMANCE SYSTEM")
        print("=" * 60)
        
        # Analyze data coverage
        coverage_stats, venue_basic_stats = self.analyze_data_coverage(df)
        
        # Primary analysis (per-dog venue effects)
        primary_effects, dog_comparisons = self.calculate_primary_venue_effects(df)
        
        # Secondary analysis (venue characteristics)
        secondary_chars = self.calculate_secondary_venue_characteristics(df)
        
        # Create combined venue performance scores
        venue_performance = {}
        
        for venue in df['venue'].unique():
            venue_data = {
                'venue': venue,
                'analysis_method': 'tertiary',  # Default
                'performance_effect': 0.0,  # Neutral default
                'confidence': 0.1,  # Low default confidence
                'sample_size': 0,
                'characteristics': {}
            }
            
            # Add secondary characteristics (always available)
            if venue in secondary_chars.index:
                char_data = secondary_chars.loc[venue]
                venue_data.update({
                    'analysis_method': 'secondary',
                    'confidence': float(char_data['data_confidence']),
                    'sample_size': int(char_data['total_races']),
                    'characteristics': {
                        'avg_field_size': float(char_data['avg_field_size']),
                        'field_difficulty': float(char_data['field_difficulty']),
                        'competitiveness': float(char_data['competitiveness']),
                        'typical_distance': char_data['typical_distance'],
                        'typical_grade': char_data['typical_grade']
                    }
                })
                
                # Use field difficulty as performance effect for secondary analysis
                # Convert to effect scale: 0.5 = neutral, >0.5 = favors stronger dogs, <0.5 = more unpredictable
                venue_data['performance_effect'] = (char_data['field_difficulty'] - 0.5) * 0.1  # Scale to ±0.05 max
            
            # Override with primary analysis if available
            if len(primary_effects) > 0 and venue in primary_effects.index:
                primary_data = primary_effects.loc[venue]
                venue_data.update({
                    'analysis_method': 'primary',
                    'performance_effect': float(primary_data['avg_effect']),
                    'confidence': float(primary_data['confidence']),
                    'sample_size': int(primary_data['comparisons'])
                })
            
            venue_performance[venue] = venue_data
        
        return venue_performance, coverage_stats
    
    def get_dog_venue_adjustment(self, dog_name: str, venue: str, venue_performance: Dict) -> Tuple[float, float]:
        """Get venue adjustment for a specific dog at a specific venue"""
        
        if venue not in venue_performance:
            return 0.0, 0.1  # Neutral effect, low confidence
        
        venue_data = venue_performance[venue]
        base_effect = venue_data['performance_effect']
        confidence = venue_data['confidence']
        
        # TODO: Add individual dog-venue history when more data becomes available
        # This would check if this specific dog has raced at this venue before
        # and use that specific performance data
        
        return base_effect, confidence
    
    def display_venue_analysis(self, venue_performance: Dict, coverage_stats: Dict):
        """Display comprehensive venue analysis results"""
        
        print(f"\n🏟️ VENUE PERFORMANCE ANALYSIS RESULTS")
        print("=" * 60)
        
        # Sort venues by confidence and effect
        sorted_venues = sorted(venue_performance.items(), 
                             key=lambda x: (x[1]['confidence'], abs(x[1]['performance_effect'])), 
                             reverse=True)
        
        print(f"{'Venue':<8} {'Effect':<8} {'Method':<10} {'Confidence':<10} {'Sample':<8} {'Field Size':<10}")
        print("-" * 70)
        
        for venue, data in sorted_venues:
            effect = f"{data['performance_effect']:+.3f}"
            method = data['analysis_method']
            confidence = f"{data['confidence']:.2f}"
            sample = str(data['sample_size'])
            field_size = f"{data['characteristics'].get('avg_field_size', 0):.1f}" if data['characteristics'] else "N/A"
            
            print(f"{venue:<8} {effect:<8} {method:<10} {confidence:<10} {sample:<8} {field_size:<10}")
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Data span: {coverage_stats['date_span_days']} days")
        print(f"   Cross-venue racing: {coverage_stats['cross_venue_percentage']:.1f}% of dogs")
        
        primary_venues = sum(1 for v in venue_performance.values() if v['analysis_method'] == 'primary')
        secondary_venues = sum(1 for v in venue_performance.values() if v['analysis_method'] == 'secondary')
        tertiary_venues = sum(1 for v in venue_performance.values() if v['analysis_method'] == 'tertiary')
        
        print(f"   Primary analysis (per-dog): {primary_venues} venues")
        print(f"   Secondary analysis (characteristics): {secondary_venues} venues") 
        print(f"   Tertiary analysis (default): {tertiary_venues} venues")
        
        print(f"\n💡 EXPANDABILITY:")
        print(f"   System ready for cross-venue data when available")
        print(f"   Will automatically upgrade analysis methods as data grows")
        print(f"   Individual dog-venue history tracking prepared")

def main():
    analyzer = RobustVenueAnalyzer()
    
    # Load and analyze data
    df = analyzer.load_comprehensive_data()
    venue_performance, coverage_stats = analyzer.create_venue_performance_system(df)
    
    # Display results
    analyzer.display_venue_analysis(venue_performance, coverage_stats)
    
    # Example usage
    print(f"\n🔍 EXAMPLE USAGE:")
    sample_venue = list(venue_performance.keys())[0]
    effect, confidence = analyzer.get_dog_venue_adjustment("Sample Dog", sample_venue, venue_performance)
    print(f"   Dog 'Sample Dog' at {sample_venue}: {effect:+.3f} effect (confidence: {confidence:.2f})")
    
    return venue_performance

if __name__ == "__main__":
    main()
