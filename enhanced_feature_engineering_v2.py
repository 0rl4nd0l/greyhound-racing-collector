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
        """Load comprehensive historical data for feature engineering with intelligent missing data handling"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            drd.id,
            drd.race_id,
            drd.dog_name,
            drd.dog_clean_name,
            drd.box_number,
            drd.finish_position,
            drd.trainer_name,
            drd.weight,
            drd.starting_price,
            drd.individual_time,
            drd.speed_rating,
            drd.recent_form,
            rm.venue,
            rm.track_condition,
            rm.weather_condition,
            rm.temperature,
            rm.distance,
            rm.grade,
            rm.race_date,
            rm.field_size,
            rm.humidity,
            rm.wind_speed
        FROM dog_race_data drd
        JOIN race_metadata rm ON drd.race_id = rm.race_id
        WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.individual_time IS NOT NULL
        ORDER BY rm.race_date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Data validation and logging
        logger.info(f"Loaded {len(df)} historical race entries")
        logger.debug(f"Sample data types: {df.dtypes}")
        
        # Clean and convert data types with robust parsing and intelligent defaults
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
        
        # Handle numeric columns with intelligent defaults
        numeric_cols_with_defaults = {
            'box_number': {'default': 4, 'range': (1, 8), 'use_median': False},
            'finish_position': {'default': None, 'range': (1, 12), 'use_median': False},  # Keep nulls for positions
            'weight': {'default': 32.0, 'range': (26.0, 38.0), 'use_median': True},  # Tighter weight range
            'starting_price': {'default': 8.0, 'range': (1.0, 100.0), 'use_median': False},  # Average odds
            'individual_time': {'default': 30.0, 'range': (26.0, 42.0), 'use_median': True},  # More realistic time range
            'speed_rating': {'default': None, 'range': (20.0, 90.0), 'use_median': False},  # Allow nulls for recalculation
            'field_size': {'default': 8, 'range': (6, 12), 'use_median': False},  # Standard field size
            'temperature': {'default': 18.0, 'range': (-5.0, 45.0), 'use_median': True},  # Melbourne average temp
            'humidity': {'default': 65.0, 'range': (20.0, 100.0), 'use_median': True},  # Average humidity
            'wind_speed': {'default': 15.0, 'range': (0.0, 50.0), 'use_median': True}  # Average wind speed
        }
        
        data_quality_report = {}
        
        for col, config in numeric_cols_with_defaults.items():
            if col in df.columns:
                # Convert to numeric, recording null count before conversion
                original_nulls = df[col].isnull().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                conversion_nulls = df[col].isnull().sum()
                
                # Apply range validation
                min_val, max_val = config['range']
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                
                # Apply intelligent defaults only where appropriate
                if config['default'] is not None:
                    # For critical fields, use defaults for out-of-range values
                    mask = (df[col].isnull() | (df[col] < min_val) | (df[col] > max_val))
                    filled_count = mask.sum()
                    
                    # Use median for fields that benefit from it, otherwise use default
                    if config.get('use_median', False) and filled_count > 0:
                        # Calculate median from valid data only
                        valid_data = df[col][(df[col] >= min_val) & (df[col] <= max_val)]
                        if len(valid_data) > 0:
                            fill_value = valid_data.median()
                            logger.info(f"  {col}: Using median {fill_value:.2f} instead of default {config['default']}")
                        else:
                            fill_value = config['default']
                            logger.info(f"  {col}: No valid data for median, using default {fill_value}")
                    else:
                        fill_value = config['default']
                    
                    df.loc[mask, col] = fill_value
                    
                    data_quality_report[col] = {
                        'original_nulls': original_nulls,
                        'conversion_nulls': conversion_nulls,
                        'out_of_range': out_of_range,
                        'filled_with_default': filled_count,
                        'default_value': config['default']
                    }
                else:
                    # For time and position data, keep nulls but log issues
                    data_quality_report[col] = {
                        'original_nulls': original_nulls,
                        'conversion_nulls': conversion_nulls,
                        'out_of_range': out_of_range,
                        'filled_with_default': 0,
                        'default_value': None
                    }
        
        # Log data quality report with reduced verbosity
        logger.info(f"Data Quality Report:")
        significant_issues = []
        for col, report in data_quality_report.items():
            total_issues = report['conversion_nulls'] + report['out_of_range']
            if total_issues > 0:
                percentage = (total_issues / len(df)) * 100
                if percentage > 5.0:  # Only log if more than 5% of data affected
                    significant_issues.append(f"{col}: {total_issues} issues ({percentage:.1f}%)")
                    if report['filled_with_default'] > 0:
                        logger.info(f"  {col}: Filled {report['filled_with_default']} values with default {report['default_value']}")
        
        if significant_issues:
            logger.warning(f"Significant data quality issues found: {'; '.join(significant_issues)}")
        else:
            logger.info(f"Data quality: Good - minimal issues detected")
        
        # Additional data cleaning
        # Remove duplicate entries (same dog, same race)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['dog_clean_name', 'race_id'], keep='first')
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate entries")
        
        # Remove races with impossible finish positions (> field_size)
        invalid_positions = df[df['finish_position'] > df['field_size']]
        if len(invalid_positions) > 0:
            logger.warning(f"Found {len(invalid_positions)} races with impossible finish positions, removing")
            df = df[df['finish_position'] <= df['field_size']]
        
        # Log final statistics
        logger.info(f"Final dataset summary:")
        logger.info(f"  Total race entries: {len(df)}")
        logger.info(f"  Unique dogs: {df['dog_clean_name'].nunique()}")
        logger.info(f"  Unique venues: {df['venue'].nunique()}")
        logger.info(f"  Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        logger.info(f"  Average field size: {df['field_size'].mean():.1f}")
        
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
        
        # 9. Distance-specific performance (required by ML model)
        distance_features = self._analyze_distance_performance(dog_history, venue)
        features.update(distance_features)
        
        # 10. Box position analysis (required by ML model)
        box_features = self._analyze_box_performance(dog_history)
        features.update(box_features)
        
        # 11. Momentum and consistency metrics (required by ML model)
        momentum_features = self._calculate_momentum_metrics(dog_history)
        features.update(momentum_features)
        
        # 12. Break quality assessment (required by ML model)
        features['break_quality'] = self._assess_break_quality(dog_history)
        
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
    
    def _analyze_distance_performance(self, dog_history, venue):
        """Analyze performance at specific distances (required by ML model)"""
        if len(dog_history) < 3:
            return {'distance_win_rate': 0, 'distance_avg_time': 30.0}
        
        # Get most common distance for this dog
        distance_performance = dog_history.groupby('distance').agg({
            'finish_position': ['mean', 'count'],
            'individual_time': 'mean'
        }).reset_index()
        
        if len(distance_performance) > 0:
            # Find best distance performance
            best_distance = distance_performance.loc[distance_performance[('finish_position', 'mean')].idxmin()]
            win_rate = (dog_history[dog_history['distance'] == best_distance['distance']]['finish_position'] == 1).mean()
            avg_time = dog_history[dog_history['distance'] == best_distance['distance']]['individual_time'].mean()
            
            return {
                'distance_win_rate': win_rate,
                'distance_avg_time': avg_time if pd.notna(avg_time) else 30.0
            }
        
        return {'distance_win_rate': 0, 'distance_avg_time': 30.0}
    
    def _analyze_box_performance(self, dog_history):
        """Analyze box position performance (required by ML model)"""
        if len(dog_history) < 3:
            return {'box_position_win_rate': 0.1, 'box_position_avg': 4.0}
        
        # Box performance analysis
        box_performance = dog_history.groupby('box_number').agg({
            'finish_position': ['mean', 'count']
        }).reset_index()
        
        if len(box_performance) > 0:
            # Overall box win rate
            box_wins = (dog_history['finish_position'] == 1).sum()
            total_races = len(dog_history)
            box_win_rate = box_wins / total_races if total_races > 0 else 0.1
            
            # Average box position performance
            avg_position = dog_history['finish_position'].mean()
            
            return {
                'box_position_win_rate': box_win_rate,
                'box_position_avg': avg_position
            }
        
        return {'box_position_win_rate': 0.1, 'box_position_avg': 4.0}
    
    def _calculate_momentum_metrics(self, dog_history):
        """Calculate momentum and consistency metrics (required by ML model)"""
        if len(dog_history) < 5:
            return {
                'recent_momentum': 0.5,
                'competitive_level': 0.5,
                'position_consistency': 0.5,
                'top_3_rate': 0.3
            }
        
        recent_positions = dog_history['finish_position'].head(10).tolist()
        
        # Recent momentum - improving/declining trend
        if len(recent_positions) >= 3:
            recent_trend = np.mean(recent_positions[:3]) - np.mean(recent_positions[3:6] if len(recent_positions) > 3 else recent_positions[:3])
            momentum = max(0, min(1, 0.5 - (recent_trend / 5)))  # Negative trend = positive momentum
        else:
            momentum = 0.5
        
        # Competitive level based on starting prices
        if 'starting_price' in dog_history.columns:
            avg_odds = dog_history['starting_price'].head(10).mean()
            competitive_level = max(0, min(1, (10 - avg_odds) / 10)) if pd.notna(avg_odds) else 0.5
        else:
            competitive_level = 0.5
        
        # Position consistency
        position_std = np.std(recent_positions[:5]) if len(recent_positions) >= 5 else 2.0
        consistency = max(0, min(1, (3.0 - position_std) / 3.0))
        
        # Top 3 rate
        top_3_count = sum(1 for pos in recent_positions[:10] if pos <= 3)
        top_3_rate = top_3_count / min(len(recent_positions), 10)
        
        return {
            'recent_momentum': momentum,
            'competitive_level': competitive_level,
            'position_consistency': consistency,
            'top_3_rate': top_3_rate
        }
    
    def _assess_break_quality(self, dog_history):
        """Assess break quality (required by ML model)"""
        if len(dog_history) < 3:
            return 0.5
        
        # Use box number as proxy for break quality (inside boxes typically have better breaks)
        recent_boxes = dog_history['box_number'].head(5).tolist()
        
        if recent_boxes:
            avg_box = np.mean([box for box in recent_boxes if pd.notna(box)])
            # Convert to 0-1 scale where lower box numbers = better breaks
            break_quality = max(0, min(1, (5 - avg_box) / 4))
            return break_quality
        
        return 0.5
    
    def _create_default_features(self):
        """Create default features for dogs with insufficient history"""
        return {
            'weighted_recent_form': 5.0,
            'speed_trend': 0,
            'speed_consistency': 1.0,
            'venue_win_rate': 0,
            'venue_avg_position': 5.0,
            'venue_experience': 0,
            'distance_win_rate': 0,
            'distance_avg_time': 30.0,
            'box_position_win_rate': 0.1,
            'box_position_avg': 4.0,
            'recent_momentum': 0.5,
            'competitive_level': 0.5,
            'position_consistency': 0.5,
            'top_3_rate': 0.3,
            'break_quality': 0.5,
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
