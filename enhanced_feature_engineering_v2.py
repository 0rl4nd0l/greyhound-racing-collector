#!/usr/bin/env python3
"""
Enhanced Feature Engineering v2.0
==================================

Advanced feature engineering specifically designed to improve prediction accuracy
by capturing greyhound racing nuances that standard ML approaches miss.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
import sqlite3
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.venue_stats = {}
        self.track_characteristics = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
    def load_comprehensive_data(self):
        """Load comprehensive historical data for feature engineering"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            drd.id,
            drd.race_id,
            drd.dog_name,
            drd.dog_clean_name,
            CAST(drd.box_number AS INTEGER) as box_number,
            CAST(drd.finish_position AS INTEGER) as finish_position,
            drd.trainer_name,
            CAST(drd.weight AS REAL) as weight,
            CAST(drd.starting_price AS REAL) as starting_price,
            CAST(drd.individual_time AS REAL) as individual_time,
            CAST(drd.speed_rating AS REAL) as speed_rating,
            drd.recent_form,
            rm.venue,
            rm.track_condition,
            rm.weather_condition,
            CAST(rm.temperature AS REAL) as temperature,
            rm.distance,
            rm.grade,
            rm.race_date,
            CAST(rm.field_size AS INTEGER) as field_size,
            CAST(rm.humidity AS REAL) as humidity,
            CAST(rm.wind_speed AS REAL) as wind_speed
        FROM dog_race_data drd
        JOIN race_metadata rm ON drd.race_id = rm.race_id
        WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.individual_time IS NOT NULL
            AND drd.finish_position REGEXP '^[0-9]+$'
            AND drd.individual_time REGEXP '^[0-9.]+$'
        ORDER BY rm.race_date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Data validation and logging
        logger.info(f"Loaded {len(df)} historical race entries")
        logger.debug(f"Sample data types: {df.dtypes}")
        
        # Clean and convert data types with robust date parsing
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['finish_position', 'box_number', 'weight', 'individual_time',
                       'starting_price', 'speed_rating', 'field_size', 'temperature',
                       'humidity', 'wind_speed']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Column {col} has {null_count} null values after conversion")
        
        # Log basic statistics
        logger.info(f"Data summary:")
        logger.info(f"  Unique dogs: {df['dog_clean_name'].nunique()}")
        logger.info(f"  Unique venues: {df['venue'].nunique()}")
        logger.info(f"  Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        
        return df
    
    def calculate_venue_specific_features(self, df):
        """Calculate venue-specific performance indicators"""
        venue_features = {}
        
        for venue in df['venue'].unique():
            venue_data = df[df['venue'] == venue]
            
            # Track bias analysis
            inside_boxes = venue_data[venue_data['box_number'] <= 4]
            outside_boxes = venue_data[venue_data['box_number'] > 4]
            
            inside_win_rate = (inside_boxes['finish_position'] == 1).mean()
            outside_win_rate = (outside_boxes['finish_position'] == 1).mean()
            
            # Speed ratings by box
            box_speed = {}
            for box in range(1, 9):
                box_data = venue_data[venue_data['box_number'] == box]
                if len(box_data) > 5:
                    box_speed[box] = box_data['individual_time'].mean()
            
            venue_features[venue] = {
                'track_bias': inside_win_rate - outside_win_rate,
                'avg_winning_time': venue_data[venue_data['finish_position'] == 1]['individual_time'].mean(),
                'field_size_impact': venue_data.groupby('field_size')['individual_time'].mean().to_dict(),
                'box_speed_ratings': box_speed,
                'weather_impact': self._calculate_weather_impact(venue_data)
            }
        
        self.venue_stats = venue_features
        return venue_features
    
    def _calculate_weather_impact(self, venue_data):
        """Calculate weather impact on performance at specific venue"""
        weather_impact = {}
        
        for weather in venue_data['weather_condition'].dropna().unique():
            weather_races = venue_data[venue_data['weather_condition'] == weather]
            if len(weather_races) > 10:
                avg_time = weather_races['individual_time'].mean()
                winner_odds = weather_races[weather_races['finish_position'] == 1]['starting_price'].mean()
                
                weather_impact[weather] = {
                    'avg_time_impact': avg_time,
                    'avg_winner_odds': winner_odds,
                    'sample_size': len(weather_races)
                }
        
        return weather_impact
    
    def create_advanced_dog_features(self, df, dog_name, race_date, venue=None):
        """Create advanced features for a specific dog"""
        # Parse race_date with proper error handling using infer_datetime_format
        try:
            if isinstance(race_date, str):
                # Use pandas automatic format detection for robust parsing
                parsed_race_date = pd.to_datetime(race_date)
            else:
                parsed_race_date = pd.to_datetime(race_date)
        except Exception as e:
            logger.warning(f"Date parsing error for race_date '{race_date}': {e}")
            parsed_race_date = pd.to_datetime('today')  # Fallback to today
        
        dog_history = df[
            (df['dog_clean_name'] == dog_name) & 
            (df['race_date'] < parsed_race_date)
        ].sort_values('race_date', ascending=False)
        
        if len(dog_history) < 3:
            return self._create_default_features()
        
        features = {}
        
        # 1. Form-based features with decay
        recent_positions = dog_history['finish_position'].head(5).tolist()
        weights = [0.4, 0.3, 0.2, 0.1, 0.05][:len(recent_positions)]
        features['weighted_recent_form'] = sum(pos * weight for pos, weight in zip(recent_positions, weights))
        
        # 2. Speed progression analysis
        recent_times = dog_history['individual_time'].head(5).dropna()
        if len(recent_times) >= 3:
            features['speed_trend'] = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
            features['speed_consistency'] = recent_times.std()
        else:
            features['speed_trend'] = 0
            features['speed_consistency'] = 0
        
        # 3. Venue-specific performance
        if venue and venue in dog_history['venue'].values:
            venue_history = dog_history[dog_history['venue'] == venue]
            features['venue_win_rate'] = (venue_history['finish_position'] == 1).mean()
            features['venue_avg_position'] = venue_history['finish_position'].mean()
            features['venue_experience'] = len(venue_history)
        else:
            features['venue_win_rate'] = 0
            features['venue_avg_position'] = 5.0
            features['venue_experience'] = 0
        
        # 4. Class progression
        grade_progression = self._analyze_grade_progression(dog_history)
        features.update(grade_progression)
        
        # 5. Break patterns
        break_analysis = self._analyze_break_patterns(dog_history)
        features.update(break_analysis)
        
        # 6. Trainer/Jockey effects (if available)
        features['trainer_impact'] = self._calculate_trainer_impact(dog_history)
        
        # 7. Seasonal performance
        features['seasonal_performance'] = self._calculate_seasonal_performance(dog_history, parsed_race_date)
        
        # 8. Competitive level analysis
        features['competition_level'] = self._analyze_competition_level(dog_history)
        
        return features
    
    def _analyze_grade_progression(self, dog_history):
        """Analyze if dog is moving up/down in class"""
        grades = dog_history['grade'].head(10).tolist()
        
        # Simple grade mapping (would need venue-specific mapping in reality)
        grade_values = {
            'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4, 'Grade 5': 5,
            'Maiden': 6, 'Mixed': 3.5
        }
        
        numeric_grades = [grade_values.get(grade, 3.5) for grade in grades if pd.notna(grade)]
        
        if len(numeric_grades) >= 3:
            recent_trend = np.mean(numeric_grades[:3]) - np.mean(numeric_grades[3:6] if len(numeric_grades) > 3 else numeric_grades[:3])
            return {
                'grade_trend': recent_trend,  # Negative = moving up in class
                'current_class_comfort': self._calculate_class_comfort(numeric_grades)
            }
        
        return {'grade_trend': 0, 'current_class_comfort': 0.5}
    
    def _calculate_class_comfort(self, numeric_grades):
        """Calculate how comfortable dog is at current class level"""
        if len(numeric_grades) < 3:
            return 0.5
        
        current_class = numeric_grades[0]
        recent_classes = numeric_grades[:5]
        
        # How often does dog run at this class level?
        class_frequency = sum(1 for grade in recent_classes if abs(grade - current_class) < 0.5) / len(recent_classes)
        
        return class_frequency
    
    def _analyze_break_patterns(self, dog_history):
        """Analyze performance after breaks"""
        if len(dog_history) < 5:
            return {'break_impact': 0, 'consistency_after_break': 0.5}
        
        # Calculate days between races
        dates = pd.to_datetime(dog_history['race_date'].head(10))
        breaks = dates.diff().dt.days.abs()
        positions = dog_history['finish_position'].head(10).tolist()
        
        # Performance after different break lengths
        short_break_performance = []  # 1-7 days
        medium_break_performance = []  # 8-21 days
        long_break_performance = []   # 22+ days
        
        for i, (break_days, position) in enumerate(zip(breaks[1:], positions[1:])):
            if pd.notna(break_days) and pd.notna(position):
                if break_days <= 7:
                    short_break_performance.append(position)
                elif break_days <= 21:
                    medium_break_performance.append(position)
                else:
                    long_break_performance.append(position)
        
        # Calculate impact
        baseline_performance = np.mean(positions)
        
        break_impact = 0
        if short_break_performance:
            break_impact += (baseline_performance - np.mean(short_break_performance)) * 0.5
        if medium_break_performance:
            break_impact += (baseline_performance - np.mean(medium_break_performance)) * 0.3
        if long_break_performance:
            break_impact += (baseline_performance - np.mean(long_break_performance)) * 0.2
        
        return {
            'break_impact': break_impact,
            'consistency_after_break': 1 / (np.std(positions[:5]) + 1) if len(positions) >= 5 else 0.5
        }
    
    def _calculate_trainer_impact(self, dog_history):
        """Calculate trainer impact on performance"""
        if 'trainer_name' not in dog_history.columns:
            return 0.5
        
        trainer_performance = dog_history.groupby('trainer_name')['finish_position'].mean()
        if len(trainer_performance) > 0:
            best_trainer_avg = trainer_performance.min()
            return max(0, (5 - best_trainer_avg) / 5)  # Normalize to 0-1
        
        return 0.5
    
    def _calculate_seasonal_performance(self, dog_history, race_date):
        """Calculate seasonal performance patterns"""
        try:
            # Ensure race_date is a datetime object
            if isinstance(race_date, str):
                race_date = pd.to_datetime(race_date)
            elif not isinstance(race_date, pd.Timestamp):
                race_date = pd.to_datetime(race_date)
            
            current_month = race_date.month
        except Exception as e:
            logger.warning(f"Date parsing error for race_date '{race_date}': {e}")
            return 0.5
        
        # Group by month
        monthly_performance = dog_history.groupby(dog_history['race_date'].dt.month)['finish_position'].mean()
        
        if current_month in monthly_performance.index:
            current_month_avg = monthly_performance[current_month]
            overall_avg = dog_history['finish_position'].mean()
            
            return max(0, (overall_avg - current_month_avg) / overall_avg)
        
        return 0.5
    
    def _analyze_competition_level(self, dog_history):
        """Analyze the level of competition dog typically faces"""
        # Use field size and average starting prices as proxy for competition level
        if len(dog_history) < 5:
            return 0.5
        
        avg_field_size = dog_history['field_size'].mean()
        avg_winner_odds = dog_history.groupby('race_id')['starting_price'].min().mean()
        
        # Normalize competition level (larger fields and lower winner odds = tougher competition)
        competition_score = (avg_field_size / 8) * 0.6 + (max(0, 5 - avg_winner_odds) / 5) * 0.4
        
        return min(1.0, competition_score)
    
    def _create_default_features(self):
        """Create default features for dogs with insufficient history"""
        return {
            'weighted_recent_form': 5.0,
            'speed_trend': 0,
            'speed_consistency': 1.0,
            'venue_win_rate': 0,
            'venue_avg_position': 5.0,
            'venue_experience': 0,
            'grade_trend': 0,
            'current_class_comfort': 0.5,
            'break_impact': 0,
            'consistency_after_break': 0.5,
            'trainer_impact': 0.5,
            'seasonal_performance': 0.5,
            'competition_level': 0.5
        }
    
    def create_race_context_features(self, race_data):
        """Create features based on race context"""
        features = {}
        
        venue = race_data.get('venue')
        track_condition = race_data.get('track_condition', 'Good')
        weather = race_data.get('weather_condition', 'Clear')
        distance = race_data.get('distance', '500m')
        field_size = race_data.get('field_size', 8)
        
        # Venue-specific adjustments
        if venue in self.venue_stats:
            venue_info = self.venue_stats[venue]
            features['track_bias_adjustment'] = venue_info.get('track_bias', 0)
            features['venue_speed_rating'] = venue_info.get('avg_winning_time', 30.0)
            
            # Weather impact at this venue
            if weather in venue_info.get('weather_impact', {}):
                weather_impact = venue_info['weather_impact'][weather]
                features['weather_time_impact'] = weather_impact.get('avg_time_impact', 0)
            else:
                features['weather_time_impact'] = 0
        else:
            features['track_bias_adjustment'] = 0
            features['venue_speed_rating'] = 30.0
            features['weather_time_impact'] = 0
        
        # Distance impact
        distance_num = float(distance.replace('m', '')) if 'm' in distance else 500
        features['distance_category'] = min(1.0, distance_num / 800)  # Normalize to 0-1
        
        # Field size impact
        features['field_size_pressure'] = min(1.0, field_size / 8)
        
        return features

def main():
    """Demonstration of enhanced feature engineering"""
    engineer = AdvancedFeatureEngineer()
    
    print("🔧 Loading comprehensive data...")
    df = engineer.load_comprehensive_data()
    
    if len(df) < 100:
        print("❌ Insufficient data for feature engineering")
        return
    
    print("📊 Calculating venue-specific features...")
    venue_features = engineer.calculate_venue_specific_features(df)
    
    print("✅ Enhanced feature engineering complete!")
    print(f"📈 Analyzed {len(df)} races across {len(venue_features)} venues")
    
    # Example: Create features for a specific dog
    if len(df) > 0:
        sample_dog = df['dog_clean_name'].iloc[0]
        sample_date = df['race_date'].iloc[0]
        sample_venue = df['venue'].iloc[0]
        
        print(f"\n🐕 Sample features for {sample_dog}:")
        features = engineer.create_advanced_dog_features(df, sample_dog, sample_date, sample_venue)
        
        for feature, value in features.items():
            print(f"  {feature}: {value:.3f}")

if __name__ == "__main__":
    main()
