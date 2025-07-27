#!/usr/bin/env python3
"""
Corrected Venue Performance Analysis
====================================

This script implements the proper method for analyzing venue performance:
- Compare each dog's performance at specific venues vs their overall performance
- Identify venues where dogs consistently perform better/worse than expected
- Measure actual track characteristics rather than meaningless averages

Author: AI Assistant  
Date: July 24, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict

class CorrectVenueAnalyzer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        
    def load_race_data(self):
        """Load race data with proper filtering"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            d.dog_name,
            d.finish_position,
            r.venue,
            r.field_size,
            r.race_date,
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
        
        print(f"📊 Loaded {len(df)} race entries for venue analysis")
        return df
    
    def calculate_per_dog_venue_performance(self, df):
        """Calculate how each dog performs at each venue vs their overall performance"""
        
        print("🔍 Calculating per-dog venue performance...")
        
        # Calculate each dog's overall performance metrics
        dog_overall_stats = df.groupby('dog_name').agg({
            'finish_position': ['mean', 'count'],
            'field_size': 'mean'
        }).reset_index()
        
        dog_overall_stats.columns = ['dog_name', 'overall_avg_position', 'total_races', 'avg_field_size']
        
        # Calculate performance score (normalized 0-1, higher = better)
        dog_overall_stats['overall_performance_score'] = (
            dog_overall_stats['avg_field_size'] - dog_overall_stats['overall_avg_position'] + 1
        ) / dog_overall_stats['avg_field_size']
        
        # Calculate each dog's performance at each venue
        dog_venue_stats = df.groupby(['dog_name', 'venue']).agg({
            'finish_position': ['mean', 'count'],
            'field_size': 'mean'
        }).reset_index()
        
        dog_venue_stats.columns = ['dog_name', 'venue', 'venue_avg_position', 'venue_races', 'venue_avg_field_size']
        
        # Calculate venue-specific performance score
        dog_venue_stats['venue_performance_score'] = (
            dog_venue_stats['venue_avg_field_size'] - dog_venue_stats['venue_avg_position'] + 1
        ) / dog_venue_stats['venue_avg_field_size']
        
        # Merge to compare venue vs overall performance
        comparison = dog_venue_stats.merge(dog_overall_stats, on='dog_name')
        
        # Calculate venue effect (positive = dog performs better at this venue)
        comparison['venue_effect'] = comparison['venue_performance_score'] - comparison['overall_performance_score']
        comparison['position_difference'] = comparison['overall_avg_position'] - comparison['venue_avg_position']
        
        return comparison
    
    def analyze_venue_characteristics(self, comparison_df):
        """Analyze actual venue characteristics based on per-dog performance differences"""
        
        print("🏟️ Analyzing venue characteristics...")
        
        # Only include dogs with multiple races (3+) at a venue for reliability
        reliable_data = comparison_df[comparison_df['venue_races'] >= 3].copy()
        
        venue_analysis = reliable_data.groupby('venue').agg({
            'venue_effect': ['mean', 'std', 'count'],
            'position_difference': ['mean', 'std'],
            'venue_races': 'sum',
            'dog_name': 'nunique'
        }).reset_index()
        
        # Flatten column names
        venue_analysis.columns = [
            'venue', 'avg_venue_effect', 'venue_effect_std', 'dogs_analyzed',
            'avg_position_improvement', 'position_improvement_std',
            'total_races', 'unique_dogs'
        ]
        
        # Calculate confidence in venue effect (more data = higher confidence)
        venue_analysis['confidence_score'] = np.minimum(
            venue_analysis['dogs_analyzed'] / 20,  # Normalize by 20 dogs
            1.0
        )
        
        # Classify venues
        venue_analysis['venue_type'] = venue_analysis['avg_venue_effect'].apply(
            lambda x: 'Favors Strong Dogs' if x > 0.02 
                     else 'Favors Weak Dogs' if x < -0.02 
                     else 'Neutral'
        )
        
        return venue_analysis.sort_values('avg_venue_effect', ascending=False)
    
    def identify_venue_specialists(self, comparison_df):
        """Identify dogs that are particularly good/bad at specific venues"""
        
        print("🎯 Identifying venue specialists...")
        
        # Filter for dogs with significant venue effects
        specialists = comparison_df[
            (comparison_df['venue_races'] >= 3) &  # At least 3 races at venue
            (comparison_df['total_races'] >= 8) &   # At least 8 total races
            (abs(comparison_df['venue_effect']) > 0.1)  # Significant effect
        ].copy()
        
        specialists['specialist_type'] = specialists['venue_effect'].apply(
            lambda x: 'Venue Specialist' if x > 0.1 else 'Venue Struggles'
        )
        
        return specialists.sort_values('venue_effect', ascending=False)
    
    def run_analysis(self):
        """Run complete corrected venue analysis"""
        
        print("🎯 CORRECTED VENUE PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Load data
        df = self.load_race_data()
        
        if len(df) == 0:
            print("❌ No data available for analysis")
            return
        
        # Calculate per-dog venue performance
        comparison = self.calculate_per_dog_venue_performance(df)
        
        # Analyze venue characteristics
        venue_analysis = self.analyze_venue_characteristics(comparison)
        
        # Display results
        print(f"\n🏟️ VENUE CHARACTERISTICS ANALYSIS")
        print(f"=" * 40)
        print(f"{'Venue':<8} {'Effect':<8} {'Type':<18} {'Dogs':<5} {'Confidence':<10}")
        print("-" * 60)
        
        for _, venue in venue_analysis.head(15).iterrows():
            effect = f"{venue['avg_venue_effect']:+.3f}"
            venue_type = venue['venue_type']
            dogs = int(venue['dogs_analyzed'])
            confidence = f"{venue['confidence_score']:.2f}"
            
            print(f"{venue['venue']:<8} {effect:<8} {venue_type:<18} {dogs:<5} {confidence:<10}")
        
        # Identify specialists
        specialists = self.identify_venue_specialists(comparison)
        
        if len(specialists) > 0:
            print(f"\n🎯 VENUE SPECIALISTS")
            print(f"=" * 30)
            print("Dogs that perform significantly better/worse at specific venues:")
            print()
            
            for _, spec in specialists.head(10).iterrows():
                dog = spec['dog_name']
                venue = spec['venue']
                effect = spec['venue_effect']
                races = int(spec['venue_races'])
                spec_type = spec['specialist_type']
                
                print(f"• {dog} at {venue}: {effect:+.3f} effect ({races} races) - {spec_type}")
        
        return venue_analysis, specialists

def main():
    analyzer = CorrectVenueAnalyzer()
    venue_analysis, specialists = analyzer.run_analysis()
    
    print(f"\n📋 SUMMARY")
    print(f"=" * 20)
    print(f"• Analysis shows actual venue characteristics")
    print(f"• Positive effect = venue favors stronger dogs")
    print(f"• Negative effect = venue creates upsets/unpredictability")
    print(f"• This method is mathematically sound and meaningful")

if __name__ == "__main__":
    main()
