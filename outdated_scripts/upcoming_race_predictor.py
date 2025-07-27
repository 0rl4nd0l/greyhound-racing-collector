#!/usr/bin/env python3
"""
Upcoming Race Predictor
===============

This script analyzes upcoming races (without results) and makes predictions
based on historical data and form analysis.

Author: AI Assistant
Date: July 11, 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sqlite3
import json
import requests
import warnings
warnings.filterwarnings('ignore')

# Advanced ML and analysis libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, log_loss
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import enhanced feature engineering
try:
    from enhanced_feature_engineering import EnhancedFeatureEngineer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    print("⚠️ Enhanced feature engineering not available")

# Import enhanced race processor
try:
    from enhanced_race_processor_fixed import EnhancedRaceProcessor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSOR_AVAILABLE = False
    print("⚠️ Enhanced race processor not available")

# Fuzzy matching removed to prevent data hallucinations
FUZZYWUZZY_AVAILABLE = False

# Weather API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class UpcomingRacePredictor:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.upcoming_dir = Path('./upcoming_races')
        self.predictions_dir = Path('./predictions')
        
        # Create predictions directory
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Advanced features
        # Get OpenWeather API key from environment or set directly
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')  # Free API key from openweathermap.org
        self.ml_models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
        # Enhanced analysis flags
        self.use_ml_predictions = SKLEARN_AVAILABLE
        self.use_fuzzy_matching = False  # Disabled to prevent data hallucinations
        self.use_weather_data = REQUESTS_AVAILABLE and self.weather_api_key
        self.use_enhanced_features = ENHANCED_FEATURES_AVAILABLE
        self.use_enhanced_processor = ENHANCED_PROCESSOR_AVAILABLE
        
        # Initialize enhanced feature engineering
        if self.use_enhanced_features:
            self.feature_engineer = EnhancedFeatureEngineer(db_path)
        else:
            self.feature_engineer = None
        
        # Initialize enhanced race processor
        if self.use_enhanced_processor:
            self.race_processor = EnhancedRaceProcessor(db_path)
        else:
            self.race_processor = None
        
        print(f"🎯 Advanced Predictor Initialized")
        print(f"✅ ML Available: {SKLEARN_AVAILABLE}")
        print(f"✅ Enhanced Features: {ENHANCED_FEATURES_AVAILABLE}")
        print(f"✅ Enhanced Processor: {ENHANCED_PROCESSOR_AVAILABLE}")
        print(f"✅ Fuzzy Matching: Disabled (prevents hallucinations)")
        print(f"✅ Weather API: {self.use_weather_data}")
    
    def extract_form_data_from_csv(self, dog_name, df):
        """Extract form data for a dog directly from the CSV file"""
        try:
            # Find all rows for this dog (including blank dog name rows that follow)
            dog_rows = []
            found_dog = False
            # Debug output (disabled)
            # print(f"     🔍 Looking for form data for: '{dog_name}'")
            
            for idx, row in df.iterrows():
                if pd.notna(row.get('Dog Name', '')) and row['Dog Name'].strip():
                    current_dog = row['Dog Name'].strip()
                    # Extract dog name from numbered format (e.g., "1. Steely Mac" -> "Steely Mac")
                    clean_current = current_dog
                    if '. ' in current_dog:
                        clean_current = current_dog.split('. ', 1)[1].strip()
                    
                    found_dog = (clean_current.upper() == dog_name.upper())
                    # if found_dog:
                    #     print(f"     ✅ Found dog: '{clean_current}' matches '{dog_name}'")
                
                if found_dog:
                    # Check if this row has race data (including rows with NaN dog names that are form data)
                    if pd.notna(row.get('TIME')):
                        try:
                            # Try to get position from PLC first, then PIR as fallback
                            position = None
                            if pd.notna(row['PLC']) and str(row['PLC']).strip():
                                try:
                                    position = int(row['PLC'])
                                except (ValueError, TypeError):
                                    pass
                            
                            # If no PLC, try PIR (Position In Race) - this is common in form guides
                            if position is None and pd.notna(row['PIR']) and str(row['PIR']).strip():
                                try:
                                    # PIR might be like "7" or "2" - extract the final position (last digit)
                                    pir_str = str(row['PIR']).strip()
                                    if pir_str.isdigit():
                                        position = int(pir_str)
                                    elif len(pir_str) > 0 and pir_str[-1].isdigit():
                                        position = int(pir_str[-1])  # Last digit as final position
                                except (ValueError, TypeError):
                                    pass
                            
                            form_entry = {
                                'position': position,
                                'time': float(row['TIME']) if pd.notna(row['TIME']) else None,
                                'distance': int(row['DIST']) if pd.notna(row['DIST']) else None,
                                'weight': float(row['WGT']) if pd.notna(row['WGT']) else None,
                                'box': int(row['BOX']) if pd.notna(row['BOX']) else None,
                                'margin': float(row['MGN']) if pd.notna(row['MGN']) and str(row['MGN']).replace('.', '').replace('-', '').isdigit() else 0,
                                'date': row['DATE'] if pd.notna(row['DATE']) else None,
                                'track': row['TRACK'] if pd.notna(row['TRACK']) else None,
                                'grade': row['G'] if pd.notna(row['G']) else None,
                                'sectional_1': float(row['1 SEC']) if pd.notna(row['1 SEC']) and str(row['1 SEC']).replace('.', '').isdigit() else None,
                                'win_time': float(row['WIN']) if pd.notna(row['WIN']) else None,
                                'bonus_time': float(row['BON']) if pd.notna(row['BON']) else None,
                                'sex': row['Sex'],
                                'starting_price': float(row['SP']) if pd.notna(row['SP']) else None,
                                'winner_runner_up': row['W/2G'] if pd.notna(row['W/2G']) else None,
                                'position_in_race': row['PIR'] if pd.notna(row['PIR']) else None
                            }
                            dog_rows.append(form_entry)
                        except (ValueError, TypeError):
                            continue
                
                # Stop when we hit the next numbered dog
                elif found_dog and pd.notna(row.get('Dog Name', '')) and row['Dog Name'].strip() and any(char.isdigit() for char in row['Dog Name'][:3]):
                    break
            
            return dog_rows
            
        except Exception as e:
            print(f"Error extracting form data for {dog_name}: {e}")
            return []
    
    def analyze_form_data(self, form_data, dog_name):
        """Analyze extracted form data and calculate performance metrics with enhanced correlations"""
        
        # Utility functions
        def calculate_time_behind(entry):
            return entry['time'] - entry['win_time'] if entry['time'] and entry['win_time'] else None
        
        def calculate_relative_performance(entry):
            win_time = entry['win_time'] if entry['win_time'] else entry['time']
            return (entry['time'] / win_time) * 100 if win_time else None
        
        def assess_competition_quality(entry):
            return len(set(entry['winner_runner_up'].split(', '))) if entry['winner_runner_up'] else None
        
        def categorize_market_position(sp):
            if not sp or pd.isna(sp):
                return 'Unknown'
            elif sp < 2.0:
                return 'Favorite'
            elif sp < 5.0:
                return 'Contender'
            else:
                return 'Outsider'
        
        def parse_grade_details(grade):
            if not grade:
                return 'Unknown'
            grade_str = str(grade)
            if 'Maiden' in grade_str:
                return 'Maiden'
            elif 'Restricted' in grade_str:
                return 'Restricted'
            elif 'Listed' in grade_str or 'Group' in grade_str:
                return 'Stakes'
            elif 'Tier' in grade_str:
                parts = grade_str.split(' ')
                return parts[1] if len(parts) > 1 else 'Tier'    
            else:
                return 'Grade'
        
        def analyze_position_in_race(pir_data):
            """Analyze position in race (PIR) data - sectional positions throughout race"""
            if not pir_data:
                return {}
            
            try:
                # PIR format example: "3233" = 3rd at first call, 2nd at second call, etc.
                pir_str = str(pir_data)
                if len(pir_str) >= 4:
                    positions = [int(pir_str[i]) for i in range(len(pir_str))]
                    return {
                        'early_position': positions[0] if len(positions) > 0 else None,
                        'mid_position': positions[1] if len(positions) > 1 else None,
                        'late_position': positions[2] if len(positions) > 2 else None,
                        'final_position': positions[3] if len(positions) > 3 else None,
                        'position_changes': len(set(positions)),
                        'closing_strong': positions[-1] < positions[0] if len(positions) > 1 else False,
                        'led_early': positions[0] == 1 if len(positions) > 0 else False
                    }
                return {}
            except (ValueError, TypeError):
                return {}
        
        def analyze_box_draw_correlation(box_numbers, positions):
            """Analyze correlation between box draw and performance"""
            box_performance = {}
            for box, pos in zip(box_numbers, positions):
                if box and pos:
                    if box not in box_performance:
                        box_performance[box] = []
                    box_performance[box].append(pos)
            
            # Calculate average position for each box drawn
            box_stats = {}
            for box, pos_list in box_performance.items():
                box_stats[box] = {
                    'avg_position': np.mean(pos_list),
                    'win_rate': sum(1 for p in pos_list if p == 1) / len(pos_list),
                    'place_rate': sum(1 for p in pos_list if p <= 3) / len(pos_list),
                    'races': len(pos_list)
                }
            return box_stats
        
        def analyze_weight_impact(weights, positions):
            """Analyze how weight changes affect performance"""
            if len(weights) < 3:
                return {'weight_sensitivity': 0, 'optimal_weight': None}
            
            # Calculate correlation between weight and position
            try:
                correlation = np.corrcoef(weights, positions)[0, 1] if len(weights) > 1 else 0
                
                # Find optimal weight range
                weight_pos_pairs = list(zip(weights, positions))
                best_performances = [w for w, p in weight_pos_pairs if p <= 3]
                optimal_weight = np.mean(best_performances) if best_performances else np.mean(weights)
                
                return {
                    'weight_sensitivity': abs(correlation),
                    'optimal_weight': optimal_weight,
                    'weight_correlation': correlation
                }
            except:
                return {'weight_sensitivity': 0, 'optimal_weight': np.mean(weights)}
        
        def analyze_margin_patterns(margins, positions):
            """Analyze winning/losing margins to assess competitiveness"""
            if not margins:
                return {'competitiveness': 0.5, 'close_finisher': False}
            
            # Count close finishes (within 2 lengths)
            close_finishes = sum(1 for m in margins if abs(m) <= 2.0)
            close_finish_rate = close_finishes / len(margins) if margins else 0
            
            # Average margin when winning vs losing
            winning_margins = [m for i, m in enumerate(margins) if i < len(positions) and positions[i] == 1]
            losing_margins = [abs(m) for i, m in enumerate(margins) if i < len(positions) and positions[i] > 1]
            
            competitiveness = min(1.0, close_finish_rate * 2)  # Higher = more competitive
            
            return {
                'competitiveness': competitiveness,
                'close_finisher': close_finish_rate > 0.3,
                'avg_winning_margin': np.mean(winning_margins) if winning_margins else 0,
                'avg_losing_margin': np.mean(losing_margins) if losing_margins else 0,
                'close_finish_rate': close_finish_rate
            }
        
        def analyze_sectional_speed_patterns(sectionals, times, positions):
            """Analyze early speed vs finishing ability"""
            if not sectionals or not times:
                return {'early_speed_advantage': 0, 'closing_ability': 0}
            
            try:
                # Calculate sectional speed (first sectional time)
                early_speeds = [s for s in sectionals if s]
                
                # Calculate closing sectionals (total time - first sectional)
                closing_sectionals = []
                for i, (sect, total) in enumerate(zip(sectionals, times)):
                    if sect and total and total > sect:
                        closing_sectionals.append(total - sect)
                
                # Correlate early speed with final position
                early_speed_positions = [(s, p) for s, p in zip(early_speeds, positions[:len(early_speeds)]) if s and p]
                if len(early_speed_positions) > 2:
                    speeds, pos = zip(*early_speed_positions)
                    early_speed_correlation = -np.corrcoef(speeds, pos)[0, 1] if len(speeds) > 1 else 0  # Negative because faster = lower time, better position
                else:
                    early_speed_correlation = 0
                
                # Analyze closing sectionals
                closing_positions = [(c, p) for c, p in zip(closing_sectionals, positions[:len(closing_sectionals)]) if c and p]
                if len(closing_positions) > 2:
                    closings, pos = zip(*closing_positions)
                    closing_correlation = -np.corrcoef(closings, pos)[0, 1] if len(closings) > 1 else 0
                else:
                    closing_correlation = 0
                
                return {
                    'early_speed_advantage': max(0, early_speed_correlation),
                    'closing_ability': max(0, closing_correlation),
                    'avg_sectional': np.mean(early_speeds) if early_speeds else 0,
                    'avg_closing': np.mean(closing_sectionals) if closing_sectionals else 0
                }
            except:
                return {'early_speed_advantage': 0, 'closing_ability': 0}
        
        def analyze_distance_performance_curve(distances, positions):
            """Analyze performance at different distances to find optimal range"""
            if not distances or len(set(distances)) < 2:
                return {'distance_versatility': 0, 'optimal_distance': None}
            
            distance_performance = {}
            for dist, pos in zip(distances, positions):
                if dist and pos:
                    if dist not in distance_performance:
                        distance_performance[dist] = []
                    distance_performance[dist].append(pos)
            
            # Calculate performance at each distance
            distance_stats = {}
            for dist, pos_list in distance_performance.items():
                if len(pos_list) >= 2:  # Minimum sample size
                    distance_stats[dist] = {
                        'avg_position': np.mean(pos_list),
                        'win_rate': sum(1 for p in pos_list if p == 1) / len(pos_list),
                        'sample_size': len(pos_list)
                    }
            
            # Find optimal distance
            if distance_stats:
                best_distance = min(distance_stats.keys(), key=lambda d: distance_stats[d]['avg_position'])
                versatility = len(distance_stats) / len(set(distances))  # How many distances performed well at
            else:
                best_distance = max(set(distances), key=distances.count) if distances else None
                versatility = 0
            
            return {
                'distance_versatility': versatility,
                'optimal_distance': best_distance,
                'distance_performance': distance_stats
            }
        if not form_data:
            return None
        
        try:
            # Extract relevant metrics
            positions = [entry['position'] for entry in form_data if entry['position'] is not None]
            times = [entry['time'] for entry in form_data if entry['time'] is not None]
            weights = [entry['weight'] for entry in form_data if entry['weight'] is not None]
            margins = [entry['margin'] for entry in form_data if entry['margin'] is not None]
            sectionals = [entry['sectional_1'] for entry in form_data if entry['sectional_1'] is not None]
            distances = [entry['distance'] for entry in form_data if entry['distance'] is not None]
            box_numbers = [entry['box'] for entry in form_data if entry['box'] is not None]
            
            # New metrics from unused data
            starting_prices = [entry['starting_price'] for entry in form_data if entry['starting_price'] is not None]
            winners_beaten = [entry['winner_runner_up'] for entry in form_data if entry['winner_runner_up'] is not None]
            pir_data = [entry['position_in_race'] for entry in form_data if entry['position_in_race'] is not None]
            sexes = [entry['sex'] for entry in form_data if entry['sex'] is not None]
            grades = [entry['grade'] for entry in form_data if entry['grade'] is not None]
            
            # Enhanced analysis of correlating factors
            box_analysis = analyze_box_draw_correlation(box_numbers, positions)
            weight_analysis = analyze_weight_impact(weights, positions)
            margin_analysis = analyze_margin_patterns(margins, positions)
            sectional_analysis = analyze_sectional_speed_patterns(sectionals, times, positions)
            distance_analysis = analyze_distance_performance_curve(distances, positions)
            
            if not positions or not times:
                return None
            
            # Calculate comprehensive statistics including new metrics
            stats = {
                'races_count': len(form_data),
                'avg_position': np.mean(positions) if positions else 0,
                'median_position': np.median(positions) if positions else 0,
                'win_rate': sum(1 for p in positions if p == 1) / len(positions) if positions else 0,
                'place_rate': sum(1 for p in positions if p <= 3) / len(positions) if positions else 0,
                'top_half_rate': sum(1 for p in positions if p <= 4) / len(positions) if positions else 0,
                'avg_time': np.mean(times) if times else 0,
                'best_time': min(times) if times else 0,
                'time_consistency': 1 / (np.std(times) + 0.1) if len(times) > 1 else 1,
                'position_consistency': 1 / (np.std(positions) + 1) if len(positions) > 1 else 1,
                'recent_form': positions[:5] if positions else [],
                'form_trend': self.calculate_form_trend(positions[:10]) if len(positions) >= 3 else 0,
                'avg_weight': np.mean(weights) if weights else 30.0,
                'avg_margin': np.mean([abs(m) for m in margins]) if margins else 0,
                'avg_sectional': np.mean(sectionals) if sectionals else 0,
                'distance_preference': max(set(distances), key=distances.count) if distances else None,
                
                # Speed ratings based on times and distances
                'speed_index': self.calculate_speed_index(form_data),
                'class_assessment': self.assess_class_from_form(form_data),
                'track_versatility': len(set(entry['track'] for entry in form_data if entry['track'])),
                'recent_activity': self.calculate_recent_activity_from_form(form_data),
                'trainer_stability': 0.5,  # Default for CSV form data (no trainer info available)
                'class_progression': 0.0,  # Default for CSV form data
                
                # Default ratings derived from performance
                'avg_performance_rating': self.derive_performance_rating(positions, times),
                'avg_speed_rating': self.derive_speed_rating(times, distances),
                'avg_class_rating': self.derive_class_rating(form_data),
                
                # NEW METRICS FROM UNUSED DATA
                # Market Intelligence (SP - Starting Price)
                'avg_starting_price': np.mean(starting_prices) if starting_prices else 10.0,
                'price_consistency': 1 / (np.std(starting_prices) + 0.1) if len(starting_prices) > 1 else 1,
                'market_confidence': self.calculate_market_confidence(starting_prices, positions),
                'value_indicator': self.calculate_value_indicator(starting_prices, positions),
                'market_position_trend': self.analyze_market_trends(starting_prices),
                
                # Competition Quality (W/2G - Winner/Runner-up)
                'competition_quality': self.analyze_competition_quality(winners_beaten),
                'quality_opponents_beaten': self.count_quality_opponents(winners_beaten, positions),
                'field_strength_faced': self.assess_field_strength(winners_beaten),
                
                # Position in Race Analysis (PIR)
                'sectional_analysis': self.analyze_sectional_positions(pir_data),
                'running_style': self.determine_running_style(pir_data),
                'tactical_ability': self.assess_tactical_ability(pir_data, positions),
                
                # Gender Analysis (Sex)
                'gender': sexes[0] if sexes else 'Unknown',
                'gender_performance_modifier': self.calculate_gender_modifier(sexes[0] if sexes else None, distances, positions),
                
                # Enhanced Grade Analysis
                'grade_distribution': self.analyze_grade_distribution(grades),
                'class_level_assessment': self.assess_class_levels(grades, positions),
                'grade_progression_detailed': self.track_grade_progression(grades, form_data),
                
                # Enhanced Time Analysis
                'time_behind_winner_avg': np.mean([calculate_time_behind(entry) for entry in form_data if calculate_time_behind(entry) is not None]),
                'relative_performance_avg': np.mean([calculate_relative_performance(entry) for entry in form_data if calculate_relative_performance(entry) is not None]),
                'closing_sectional_strength': self.analyze_closing_speed(form_data),
                
                # ENHANCED CORRELATING FACTORS FOR BETTER PREDICTIONS
                # Box Draw Analysis
                'box_draw_performance': box_analysis,
                'preferred_box_numbers': [box for box, stats in box_analysis.items() if stats['win_rate'] > 0.2] if box_analysis else [],
                'box_versatility': len(box_analysis) if box_analysis else 0,
                
                # Weight Impact Analysis
                'weight_sensitivity': weight_analysis.get('weight_sensitivity', 0),
                'optimal_weight': weight_analysis.get('optimal_weight'),
                'weight_correlation': weight_analysis.get('weight_correlation', 0),
                
                # Competitiveness & Margin Analysis
                'competitiveness_score': margin_analysis.get('competitiveness', 0.5),
                'close_finisher': margin_analysis.get('close_finisher', False),
                'avg_winning_margin': margin_analysis.get('avg_winning_margin', 0),
                'close_finish_rate': margin_analysis.get('close_finish_rate', 0),
                
                # Sectional Speed Patterns
                'early_speed_advantage': sectional_analysis.get('early_speed_advantage', 0),
                'closing_ability': sectional_analysis.get('closing_ability', 0),
                'speed_profile': 'Early' if sectional_analysis.get('early_speed_advantage', 0) > sectional_analysis.get('closing_ability', 0) else 'Closing',
                
                # Distance Specialization
                'distance_versatility': distance_analysis.get('distance_versatility', 0),
                'optimal_distance': distance_analysis.get('optimal_distance'),
                'distance_specialization': distance_analysis.get('distance_performance', {}),
                
                # Pressure/Pace Analysis
                'handles_pressure': margin_analysis.get('close_finish_rate', 0) > 0.3 and any(pos <= 3 for pos in positions),
                'front_runner': sectional_analysis.get('early_speed_advantage', 0) > 0.6,
                'come_from_behind': sectional_analysis.get('closing_ability', 0) > 0.6,
                
                # Raw data for ML features
                'raw_results': form_data
            }
            
            return stats
        except Exception as e:
            print(f"Error analyzing form data for {dog_name}: {e}")
            return None

    def calculate_market_confidence(self, starting_prices, positions):
        """Calculate market confidence from starting prices and positions"""
        if not starting_prices:
            return 0
        try:
            # Confidence is higher for stable markets with many favorites performing well
            volatility = np.std(starting_prices)
            favored_positions = sum(1 for sp, pos in zip(starting_prices, positions) if sp < 5 and pos <= 3)
            return max(0, min(1, 0.5 - volatility + 0.3 * favored_positions / len(starting_prices)))
        except:
            return 0

    def analyze_competition_quality(self, winners_beaten):
        """Analyze the quality of competition based on winners beaten"""
        if not winners_beaten:
            return 0
        return len(set(winners_beaten))

    def analyze_sectional_positions(self, pir_data):
        """Analyze sectional positions based on position-in-race data"""
        try:
            avg_positions = [np.mean([int(p[i]) for p in pir_data if len(p) > i]) for i in range(4)]
            return avg_positions
        except:
            return [0, 0, 0, 0]

    def determine_running_style(self, pir_data):
        """Determine running style based on average early and late positions"""
        try:
            early_positions = [int(p[0]) for p in pir_data if len(p) > 0]
            late_positions = [int(p[2]) for p in pir_data if len(p) > 2]
            if not early_positions or not late_positions:
                return 'Unknown'
            avg_early = np.mean(early_positions)
            avg_late = np.mean(late_positions)
            if avg_early == avg_late:
                return 'Consistent'
            elif avg_late < avg_early:
                return 'Finisher'
            else:
                return 'Fader'
        except:
            return 'Unknown'

    def calculate_gender_modifier(self, gender, distances, positions):
        """Calculate a gender performance modifier"""
        if not gender or not positions:
            return 1.0
        try:
            if gender.lower() == 'female':
                favored_distance = max(set(distances), key=distances.count)
                favored_rate = sum(1 for d, p in zip(distances, positions) if d == favored_distance and p <3) / len(distances)
                return 1.05 if favored_rate > 0.3 else 1.0
            return 1.0
        except:
            return 1.0

    def calculate_value_indicator(self, starting_prices, positions):
        """Calculate a value indicator based on starting prices and performance"""
        try:
            underpriced_success_rate = sum(1 for sp, pos in zip(starting_prices, positions) if sp > 5 and pos <3) / len(starting_prices)
            return max(0, min(1, underpriced_success_rate))
        except:
            return 0

    def analyze_market_trends(self, starting_prices):
        """Analyze market trends from starting prices"""
        try:
            if len(starting_prices) > 5:
                trend = np.polyfit(range(len(starting_prices)), starting_prices, 1)[0]
                return trend
            return 0
        except:
            return 0

    def count_quality_opponents(self, winners_beaten, positions):
        """Count quality opponents beaten"""
        return sum(1 for w in winners_beaten if w and positions[winners_beaten.index(w)] <3)

    def assess_field_strength(self, winners_beaten):
        """Assess the field strength"""
        try:
            return len(set(winners_beaten)) * 0.1
        except:
            return 0

    def assess_tactical_ability(self, pir_data, positions):
        """Assess tactical ability based on PIR and final positions"""
        try:
            improvements = sum(p[-1] < p[0] for p in pir_data if len(p) > 3)
            return improvements / len(positions) if positions else 0
        except:
            return 0

    def analyze_grade_distribution(self, grades):
        """Analyze grade distribution"""
        return {g: grades.count(g) / len(grades) for g in set(grades)}

    def assess_class_levels(self, grades, positions):
        """Assess class levels relative to positions"""
        try:
            return sum(p < 3 for p in positions) / len(positions)
        except:
            return 0

    def track_grade_progression(self, grades, form_data):
        """Track grade progression over form data"""
        try:
            return len(set(grades)) / len(form_data)
        except:
            return 0

    def analyze_closing_speed(self, form_data):
        """Analyze closing speed from sectional times"""
        try:
            sectional_times = [entry['sectional_1'] for entry in form_data if entry['sectional_1']]
            overall_times = [entry['time'] for entry in form_data if entry['time']]
            closing_speeds = [o - s for s, o in zip(sectional_times, overall_times) if o and s]
            return np.mean(closing_speeds) if closing_speeds else 0
        except:
            return 0
    
    def calculate_speed_index(self, form_data):
        """Calculate speed index based on times and distances"""
        try:
            if not form_data:
                return 50.0
            
            speed_ratings = []
            for entry in form_data:
                if entry.get('time') and entry.get('distance'):
                    # Simple speed calculation (distance/time)
                    speed = entry['distance'] / entry['time']
                    speed_ratings.append(speed)
            
            if speed_ratings:
                avg_speed = np.mean(speed_ratings)
                # Normalize to 0-100 scale (rough approximation)
                return min(100, max(0, (avg_speed - 15) * 5))
            
            return 50.0
        except:
            return 50.0
    
    def assess_class_from_form(self, form_data):
        """Assess class level from form data"""
        try:
            if not form_data:
                return 50.0
            
            # Extract grades and assess quality
            grades = [entry.get('grade', '') for entry in form_data]
            grade_scores = []
            
            for grade in grades:
                if not grade:
                    continue
                
                grade_str = str(grade).upper()
                
                # Handle specific grade classifications
                if '5' in grade_str:
                    grade_scores.append(90)  # Grade 5 is high class
                elif '4' in grade_str:
                    grade_scores.append(70)
                elif '3' in grade_str:
                    grade_scores.append(50)
                elif '2' in grade_str:
                    grade_scores.append(40)
                elif '1' in grade_str:
                    grade_scores.append(35)
                elif grade_str in ['MAIDEN', 'MDN', 'M', 'MAID']:
                    grade_scores.append(25)  # Maiden races - lower class for beginners
                elif grade_str in ['NV', 'NOV', 'NOVICE']:
                    grade_scores.append(30)  # Novice - slightly above maiden
                elif grade_str in ['NG', 'NO GRADE', 'NOGRADE']:
                    grade_scores.append(30)  # No grading
                elif 'FFA' in grade_str or 'FREE' in grade_str:
                    grade_scores.append(95)  # Free For All - highest class
                elif 'GROUP' in grade_str:
                    grade_scores.append(100)  # Group races - elite level
                else:
                    grade_scores.append(50)  # Default for unknown grades
            
            return np.mean(grade_scores) if grade_scores else 50.0
        except:
            return 50.0
    
    def calculate_recent_activity_from_form(self, form_data):
        """Calculate recent activity from form data dates"""
        try:
            if not form_data:
                return {'days_since_last_race': 365, 'recent_frequency': 0, 'activity_score': 0}
            
            dates = []
            for entry in form_data:
                if entry.get('date'):
                    try:
                        date_obj = pd.to_datetime(entry['date'])
                        dates.append(date_obj)
                    except:
                        continue
            
            if not dates:
                return {'days_since_last_race': 365, 'recent_frequency': 0, 'activity_score': 0}
            
            dates.sort(reverse=True)  # Most recent first
            now = pd.Timestamp.now()
            
            # Days since last race
            days_since_last = (now - dates[0]).days if dates else 365
            
            # Recent frequency (races in last 6 months)
            six_months_ago = now - pd.Timedelta(days=180)
            recent_races = sum(1 for date in dates if date >= six_months_ago)
            frequency = recent_races / 6.0
            
            # Activity score
            activity_score = max(0, min(1, (60 - days_since_last) / 60)) * min(1, frequency / 2)
            
            return {
                'days_since_last_race': days_since_last,
                'recent_frequency': frequency,
                'activity_score': activity_score
            }
        except:
            return {'days_since_last_race': 30, 'recent_frequency': 1, 'activity_score': 0.5}
    
    def derive_performance_rating(self, positions, times):
        """Derive performance rating from positions and times"""
        try:
            if not positions:
                return 50.0
            
            # Position-based rating (lower positions = higher rating)
            avg_position = np.mean(positions)
            position_rating = max(0, min(100, (8 - avg_position) * 12.5))
            
            # Time consistency bonus
            if len(times) > 1:
                time_std = np.std(times)
                consistency_bonus = max(0, 10 - time_std)
                position_rating += consistency_bonus
            
            return min(100, position_rating)
        except:
            return 50.0
    
    def derive_speed_rating(self, times, distances):
        """Derive speed rating from times and distances"""
        try:
            if not times or not distances:
                return 50.0
            
            speed_ratings = []
            for i, time in enumerate(times):
                if i < len(distances) and distances[i] and time:
                    speed = distances[i] / time
                    # Normalize speed to 0-100 scale
                    normalized = (speed - 15) * 4  # Rough normalization
                    speed_ratings.append(max(0, min(100, normalized)))
            
            return np.mean(speed_ratings) if speed_ratings else 50.0
        except:
            return 50.0
    
    def derive_class_rating(self, form_data):
        """Derive class rating from form data"""
        try:
            return self.assess_class_from_form(form_data)
        except:
            return 50.0
    
    def get_comprehensive_dog_performance(self, dog_name, df=None):
        """Get comprehensive historical performance data with advanced metrics"""
        # First try to get data from CSV form if available
        if df is not None:
            form_data = self.extract_form_data_from_csv(dog_name, df)
            if form_data:
                return self.analyze_form_data(form_data, dog_name)
        
        # Fallback to database query
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Enhanced query with venue, weather, and track conditions
            cursor.execute("""
                SELECT 
                    drd.finish_position,
                    drd.individual_time,
                    drd.margin,
                    drd.starting_price,
                    drd.performance_rating,
                    drd.speed_rating,
                    drd.class_rating,
                    drd.win_probability,
                    drd.place_probability,
                    drd.weight,
                    drd.box_number,
                    drd.trainer_name,
                    drd.recent_form,
                    drd.best_time,
                    rm.venue,
                    rm.race_date,
                    rm.distance,
                    rm.track_condition,
                    rm.weather,
                    rm.temperature,
                    rm.field_size,
                    rm.grade
                FROM dog_race_data drd
                JOIN race_metadata rm ON drd.race_id = rm.race_id
                WHERE drd.dog_clean_name = ? 
                AND drd.finish_position IS NOT NULL 
                AND drd.finish_position != ''
                ORDER BY rm.race_date DESC, drd.extraction_timestamp DESC
                LIMIT 20
            """, (dog_name.upper(),))
            
            results = cursor.fetchall()
            
            if not results:
                return None
            
            # Enhanced statistical calculations
            positions = [int(r[0]) for r in results if r[0] and str(r[0]).isdigit()]
            times = [float(r[1]) for r in results if r[1] and str(r[1]).replace('.', '').isdigit()]
            odds = [float(r[3]) for r in results if r[3] and str(r[3]).replace('.', '').replace('-', '').isdigit()]
            weights = [float(r[9]) for r in results if r[9] and str(r[9]).replace('.', '').isdigit()]
            venues = [r[14] for r in results if r[14]]
            distances = [r[16] for r in results if r[16]]
            
            # Advanced statistics
            stats = {
                'races_count': len(results),
                'avg_position': np.mean(positions) if positions else 0,
                'median_position': np.median(positions) if positions else 0,
                'win_rate': sum(1 for p in positions if p == 1) / len(positions) if positions else 0,
                'place_rate': sum(1 for p in positions if p <= 3) / len(positions) if positions else 0,
                'top_half_rate': sum(1 for p in positions if p <= 4) / len(positions) if positions else 0,
                'avg_time': np.mean(times) if times else 0,
                'best_time': min(times) if times else 0,
                'time_consistency': 1 / (np.std(times) + 0.1) if len(times) > 1 else 1,
                'position_consistency': 1 / (np.std(positions) + 1) if len(positions) > 1 else 1,
                'recent_form': positions[:5] if positions else [],
                'form_trend': self.calculate_form_trend(positions[:10]) if len(positions) >= 3 else 0,
                'avg_odds': np.mean(odds) if odds else 10.0,
                'avg_weight': np.mean(weights) if weights else 30.0,
                'venue_diversity': len(set(venues)) if venues else 0,
                'distance_preference': max(set(distances), key=distances.count) if distances else None,
                'class_progression': self.calculate_class_progression(results),
                'trainer_stability': self.calculate_trainer_stability(results),
                'seasonal_performance': self.calculate_seasonal_performance(results),
                'track_condition_performance': self.analyze_track_conditions(results),
                'recent_activity': self.calculate_recent_activity(results),
                
                # Original ratings (enhanced)
                'avg_performance_rating': np.mean([r[4] for r in results if r[4]]) if any(r[4] for r in results) else 50,
                'avg_speed_rating': np.mean([r[5] for r in results if r[5]]) if any(r[5] for r in results) else 50,
                'avg_class_rating': np.mean([r[6] for r in results if r[6]]) if any(r[6] for r in results) else 50,
                
                # Raw data for ML features
                'raw_results': results
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting historical data for {dog_name}: {e}")
            return None
        finally:
            conn.close()
    
    def calculate_form_trend(self, positions):
        """Calculate form trend (positive = improving, negative = declining)"""
        if len(positions) < 3:
            return 0
        try:
            # Use linear regression slope to determine trend
            x = np.arange(len(positions))
            y = np.array(positions)
            slope = np.polyfit(x, y, 1)[0]
            return -slope  # Negative slope (improving positions) = positive trend
        except:
            return 0
    
    def calculate_class_progression(self, results):
        """Calculate class/grade progression"""
        try:
            grades = [r[21] for r in results if r[21]]  # grade column
            if len(grades) < 2:
                return 0
            
            # Simple progression indicator (would need more sophisticated grade mapping)
            grade_changes = 0
            for i in range(1, len(grades)):
                if grades[i] != grades[i-1]:
                    grade_changes += 1
            
            return grade_changes / len(grades)  # Higher = more grade movement
        except:
            return 0
    
    def calculate_trainer_stability(self, results):
        """Calculate trainer stability (consistency)"""
        try:
            trainers = [r[11] for r in results if r[11]]  # trainer_name column
            if not trainers:
                return 0.5
            
            most_common_trainer = max(set(trainers), key=trainers.count)
            stability = trainers.count(most_common_trainer) / len(trainers)
            return stability
        except:
            return 0.5
    
    def calculate_seasonal_performance(self, results):
        """Calculate seasonal performance patterns"""
        try:
            from datetime import datetime
            race_dates = []
            positions = []
            
            for r in results:
                if r[15]:  # race_date column
                    try:
                        date = datetime.strptime(str(r[15]), '%Y-%m-%d')
                        pos = int(r[0]) if r[0] and str(r[0]).isdigit() else None
                        if pos:
                            race_dates.append(date)
                            positions.append(pos)
                    except:
                        continue
            
            if len(positions) < 5:
                return 0.5
            
            # Calculate performance by season (simplified: recent vs older)
            mid_point = len(positions) // 2
            recent_avg = np.mean(positions[:mid_point])
            older_avg = np.mean(positions[mid_point:])
            
            # Lower position is better, so improvement = older_avg > recent_avg
            improvement = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
            return max(0, min(1, 0.5 + improvement))
        except:
            return 0.5
    
    def analyze_track_conditions(self, results):
        """Analyze performance under different track conditions"""
        try:
            condition_performance = {}
            
            for r in results:
                condition = r[17] or 'Unknown'  # track_condition column
                position = int(r[0]) if r[0] and str(r[0]).isdigit() else None
                
                if position:
                    if condition not in condition_performance:
                        condition_performance[condition] = []
                    condition_performance[condition].append(position)
            
            # Calculate average position for each condition
            for condition in condition_performance:
                positions = condition_performance[condition]
                condition_performance[condition] = {
                    'avg_position': np.mean(positions),
                    'races': len(positions),
                    'win_rate': sum(1 for p in positions if p == 1) / len(positions)
                }
            
            return condition_performance
        except:
            return {}
    
    def calculate_recent_activity(self, results):
        """Calculate recent racing activity level"""
        try:
            from datetime import datetime, timedelta
            
            race_dates = []
            for r in results:
                if r[15]:  # race_date column
                    try:
                        date = datetime.strptime(str(r[15]), '%Y-%m-%d')
                        race_dates.append(date)
                    except:
                        continue
            
            if not race_dates:
                return 0
            
            race_dates.sort(reverse=True)
            now = datetime.now()
            
            # Calculate days since last race
            days_since_last = (now - race_dates[0]).days if race_dates else 365
            
            # Calculate racing frequency (races per month in last 6 months)
            six_months_ago = now - timedelta(days=180)
            recent_races = sum(1 for date in race_dates if date >= six_months_ago)
            frequency = recent_races / 6.0  # races per month
            
            return {
                'days_since_last_race': days_since_last,
                'recent_frequency': frequency,
                'activity_score': max(0, min(1, (30 - days_since_last) / 30)) * min(1, frequency / 2)
            }
        except:
            return {'days_since_last_race': 30, 'recent_frequency': 1, 'activity_score': 0.5}
    
    def get_historical_dog_performance(self, dog_name):
        """Backward compatibility - calls comprehensive method"""
        return self.get_comprehensive_dog_performance(dog_name)
    
    def calculate_enhanced_prediction_score(self, dog_stats, race_context=None):
        """Calculate enhanced prediction score using advanced features and ML"""
        if not dog_stats:
            return 0.1  # Default low score for unknown dogs
        
        # Use enhanced feature engineering if available
        if self.use_enhanced_features and self.feature_engineer:
            features = self.feature_engineer.create_advanced_features(dog_stats, race_context)
            features = self.feature_engineer.validate_features(features)
            
            # Advanced scoring using engineered features
            score = self._calculate_score_from_features(features)
        else:
            # Fallback to traditional scoring
            score = self._calculate_traditional_score(dog_stats, race_context)
        
        return min(score, 1.0)  # Cap at 100%
    
    def _calculate_score_from_features(self, features):
        """Calculate prediction score from engineered features"""
        # Weighted combination of key composite features
        score = (
            features['competitive_ability'] * 0.30 +      # Core competitive ability (30%)
            features['form_momentum'] * 0.25 +            # Current form trend (25%)
            features['fitness_score'] * 0.20 +           # Physical condition (20%)
            features['contextual_advantage'] * 0.15 +    # Race-specific advantages (15%)
            features['adaptability'] * 0.10             # Environmental adaptation (10%)
        )
        
        # Confidence boost for high-experience dogs
        if features['experience_level'] > 0.7:
            score *= 1.05  # 5% boost for experienced dogs
        
        # Penalty for poor recent form
        if features['weighted_recent_position'] > 6:
            score *= 0.9  # 10% penalty for consistently poor performance
        
        return score
    
    def _calculate_traditional_score(self, dog_stats, race_context):
        """Enhanced traditional scoring method with more correlating factors"""
        score = 0
        
        # Core performance metrics (40% total)
        score += dog_stats['win_rate'] * 0.18  # Win rate (18%)
        score += dog_stats['place_rate'] * 0.12  # Place rate (12%)
        score += dog_stats['top_half_rate'] * 0.10  # Top half rate (10%)
        
        # Position quality (12%)
        if dog_stats['avg_position'] > 0:
            position_score = (8 - dog_stats['avg_position']) / 8
            score += position_score * 0.12
        
        # Consistency and form (15% total)
        score += dog_stats['position_consistency'] * 0.08  # Position consistency (8%)
        score += dog_stats['time_consistency'] * 0.07  # Time consistency (7%)
        
        # ENHANCED CORRELATING FACTORS (18% total)
        # Competitiveness factor (5%)
        competitiveness = dog_stats.get('competitiveness_score', 0.5)
        score += competitiveness * 0.05
        
        # Box draw performance (4%)
        if race_context and 'field_size' in race_context:
            preferred_boxes = dog_stats.get('preferred_box_numbers', [])
            box_versatility = dog_stats.get('box_versatility', 0)
            if preferred_boxes or box_versatility > 2:
                score += 0.04  # Bonus for good box draw history
        
        # Speed profile matching (4%)
        early_speed = dog_stats.get('early_speed_advantage', 0)
        closing_ability = dog_stats.get('closing_ability', 0)
        speed_bonus = max(early_speed, closing_ability) * 0.04
        score += speed_bonus
        
        # Distance specialization (3%)
        if race_context and 'distance' in race_context:
            optimal_distance = dog_stats.get('optimal_distance')
            current_distance = race_context.get('distance')
            if optimal_distance and str(optimal_distance) == str(current_distance):
                score += 0.03
        
        # Close finisher bonus (2%)
        if dog_stats.get('close_finisher', False):
            score += 0.02
        
        # Form trend with enhanced weighting (8%)
        form_trend_normalized = max(0, min(1, (dog_stats['form_trend'] + 2) / 4))  # Normalize to 0-1
        score += form_trend_normalized * 0.08
        
        # Recent activity and fitness (7% total)
        activity = dog_stats.get('recent_activity', {})
        if isinstance(activity, dict):
            score += activity.get('activity_score', 0.5) * 0.07
        
        # Enhanced ratings integration with more weight (8%)
        if all(key in dog_stats for key in ['avg_performance_rating', 'avg_speed_rating', 'avg_class_rating']):
            combined_rating = (dog_stats['avg_performance_rating'] + 
                             dog_stats['avg_speed_rating'] + 
                             dog_stats['avg_class_rating']) / 3
            score += (combined_rating / 100) * 0.08  # Increased from 5% to 8%
        
        # Context-based adjustments
        if race_context:
            score = self.apply_contextual_adjustments(score, dog_stats, race_context)
        
        return score
    
    def calculate_prediction_score(self, dog_stats, race_context=None):
        """Enhanced prediction score - backward compatibility wrapper"""
        return self.calculate_enhanced_prediction_score(dog_stats, race_context)
    
    def apply_contextual_adjustments(self, base_score, dog_stats, race_context):
        """Apply race-specific contextual adjustments"""
        adjusted_score = base_score
        
        try:
            # Track condition preferences
            track_performance = dog_stats.get('track_condition_performance', {})
            current_condition = race_context.get('track_condition', 'Good')
            
            if current_condition in track_performance:
                condition_data = track_performance[current_condition]
                if condition_data['races'] >= 2:  # Minimum sample size
                    # Better than average on this condition = bonus
                    condition_advantage = (4 - condition_data['avg_position']) / 4
                    adjusted_score += condition_advantage * 0.05  # Up to 5% bonus
            
            # Distance suitability
            preferred_distance = dog_stats.get('distance_preference')
            current_distance = race_context.get('distance')
            if preferred_distance and current_distance and str(preferred_distance) == str(current_distance):
                adjusted_score += 0.03  # 3% bonus for preferred distance
            
            # Venue experience
            venue_experience = dog_stats.get('venue_diversity', 0)
            if venue_experience >= 3:  # Experienced at multiple venues
                adjusted_score += 0.02  # 2% bonus for versatility
            
            # Weather conditions (if available)
            weather_conditions = race_context.get('weather')
            if weather_conditions and 'track_condition_performance' in dog_stats:
                # Could add weather-specific adjustments here
                pass
            
        except Exception as e:
            print(f"Warning: Contextual adjustment error: {e}")
        
        return min(adjusted_score, 1.0)
    
    def _get_track_condition_from_db(self, venue, race_date):
        """Get track condition from database for venue and date"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Look for recent track conditions at this venue
            cursor.execute("""
                SELECT track_condition 
                FROM race_metadata 
                WHERE venue = ? 
                AND race_date = ?
                AND track_condition IS NOT NULL 
                AND track_condition != ''
                ORDER BY extraction_timestamp DESC
                LIMIT 1
            """, (venue, race_date))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"   ⚠️ Error getting track condition from database: {e}")
            return None
    
    def _get_venue_default_condition(self, venue):
        """Get intelligent default track condition based on venue characteristics"""
        # Venue-specific defaults based on typical conditions
        venue_defaults = {
            'AP_K': 'Good',      # Angle Park - generally good conditions
            'DAPT': 'Good',      # Dapto - coastal, can be variable
            'CASO': 'Fast',      # Casino - northern NSW, often dry
            'MEA': 'Good',       # The Meadows - Melbourne, variable
            'SAN': 'Good',       # Sandown - Melbourne, established track
            'HOBT': 'Good',      # Hobart - Tasmania, cooler climate
            'GAWL': 'Good',      # Gawler - SA, inland conditions
            'BALL': 'Good',      # Ballarat - regional Victoria
            'BEND': 'Good',      # Bendigo - regional Victoria
            'SALE': 'Good',      # Sale - Gippsland region
            'WARR': 'Good',      # Warragul - Gippsland region
            'GEELONG': 'Good',   # Geelong - coastal Victoria
            'TRARALGON': 'Good', # Traralgon - Gippsland
            'SHEPPARTON': 'Good' # Shepparton - northern Victoria
        }
        
        return venue_defaults.get(venue, 'Good')  # Default to 'Good' for unknown venues
    
    def get_race_context(self, race_info, df):
        """Get current race context for predictions"""
        from datetime import datetime
        context = {
            'venue': race_info.get('venue', 'Unknown'),
            'race_date': race_info.get('race_date', datetime.now().strftime('%Y-%m-%d')),
            'field_size': len(df)
        }
        
        # Extract RACE-SPECIFIC data (not historical data)
        if not df.empty:
            first_row = df.iloc[0]  # All dogs in same race have same race conditions
            
            # Extract DISTANCE from current race data (not historical)
            if 'DIST' in df.columns:
                dist_value = first_row.get('DIST')
                if pd.notna(dist_value) and dist_value:
                    try:
                        # Convert to string format (e.g., "520" -> "520")
                        context['distance'] = str(int(float(dist_value)))
                        print(f"   📏 Extracted race distance: {context['distance']}m")
                    except (ValueError, TypeError):
                        context['distance'] = '500'  # Fallback only if extraction fails
                        print(f"   ⚠️ Could not parse distance, using 500m default")
                else:
                    context['distance'] = '500'
                    print(f"   ⚠️ No distance data found, using 500m default")
            else:
                context['distance'] = '500'
                print(f"   ⚠️ No DIST column found, using 500m default")
            
            # Extract GRADE from current race data
            if 'G' in df.columns:
                grade_value = first_row.get('G')
                if pd.notna(grade_value) and grade_value:
                    # Handle different grade formats
                    grade_str = str(grade_value).strip()
                    
                    # Check for maiden races (various formats)
                    if grade_str.upper() in ['MAIDEN', 'MDN', 'M', 'MAID']:
                        context['grade'] = 'Maiden'
                    elif grade_str.upper() in ['NV', 'NOV', 'NOVICE']:
                        context['grade'] = 'Novice'
                    elif grade_str.upper() in ['NG', 'NO GRADE', 'NOGRADE']:
                        context['grade'] = 'No Grade'
                    else:
                        context['grade'] = grade_str
                    
                    print(f"   🏁 Extracted race grade: {context['grade']}")
            
            # Extract TRACK CONDITION - try multiple approaches
            track_condition = None
            
            # Method 1: Check if there's a track condition column in the CSV
            possible_condition_cols = ['TRACK_CONDITION', 'CONDITION', 'TRACK_COND', 'COND']
            for col in possible_condition_cols:
                if col in df.columns:
                    cond_value = first_row.get(col)
                    if pd.notna(cond_value) and cond_value and str(cond_value).lower() != 'nan':
                        track_condition = str(cond_value).title()
                        print(f"   🌤️ Extracted track condition from CSV: {track_condition}")
                        break
            
            # Method 2: Try to get from database for this venue and date
            if not track_condition:
                track_condition = self._get_track_condition_from_db(context['venue'], context['race_date'])
                if track_condition:
                    print(f"   🗄️ Found track condition in database: {track_condition}")
            
            # Method 3: Try weather API if available
            if not track_condition and self.use_weather_data:
                weather_data = self.get_current_weather(race_info.get('venue'))
                if weather_data and weather_data.get('track_condition'):
                    track_condition = weather_data['track_condition']
                    print(f"   🌤️ Derived track condition from weather: {track_condition}")
                    context.update(weather_data)
            
            # Set track condition or use intelligent default
            if track_condition:
                context['track_condition'] = track_condition
            else:
                # Use venue-based intelligent default instead of hardcoded "Good"
                context['track_condition'] = self._get_venue_default_condition(context['venue'])
                print(f"   ⚠️ No track condition found, using venue default: {context['track_condition']}")
        
        return context
    
    def enhanced_clean_dog_name(self, name):
        """Clean dog name using exact matching only (no fuzzy matching to prevent hallucinations)"""
        return self.clean_dog_name(name)
    
    # Fuzzy matching function removed to prevent data hallucinations
    # Dog names should be consistent and exact matching is more reliable
    
    def get_ml_prediction(self, dog_stats, race_context):
        """Get machine learning prediction score"""
        if not SKLEARN_AVAILABLE or not dog_stats:
            return None
        
        try:
            # Train ML model on-demand if not already trained
            if 'race_winner' not in self.ml_models:
                self.train_ml_models()
            
            # Prepare features for ML prediction
            features = self.prepare_ml_features(dog_stats, race_context)
            
            if features is not None and 'race_winner' in self.ml_models:
                model = self.ml_models['race_winner']
                scaler = self.scalers.get('race_winner')
                
                if scaler:
                    features_scaled = scaler.transform([features])
                    prediction = model.predict_proba(features_scaled)[0][1]  # Probability of winning
                    return prediction
                
        except Exception as e:
            print(f"Warning: ML prediction error: {e}")
        
        return None
    
    def combine_prediction_scores(self, traditional_score, ml_score):
        """Combine traditional and ML prediction scores"""
        if ml_score is None:
            return traditional_score
        
        # Weighted combination: 70% traditional, 30% ML
        combined = traditional_score * 0.7 + ml_score * 0.3
        return min(combined, 1.0)
    
    def get_enhanced_bet_recommendation(self, score, dog_stats):
        """Get enhanced betting recommendation"""
        confidence = self.calculate_confidence_level(dog_stats)
        
        if score >= 0.7 and confidence >= 0.8:
            return "Strong Win"
        elif score >= 0.5 and confidence >= 0.6:
            return "Win/Place"
        elif score >= 0.3 and confidence >= 0.4:
            return "Place Only"
        elif score >= 0.2:
            return "Each-Way"
        else:
            return "Avoid"
    
    def calculate_confidence_level(self, dog_stats):
        """Calculate confidence level based on data quality"""
        if not dog_stats:
            return 0.1
        
        confidence = 0.5  # Base confidence
        
        # More races = higher confidence
        races_count = dog_stats.get('races_count', 0)
        confidence += min(0.3, races_count / 20)  # Up to 30% boost for experience
        
        # Recent activity boosts confidence
        activity = dog_stats.get('recent_activity', {})
        if isinstance(activity, dict):
            confidence += activity.get('activity_score', 0) * 0.2  # Up to 20% boost
        
        return min(confidence, 1.0)
    
    def identify_key_factors(self, dog_stats, race_context):
        """Identify key factors influencing the prediction"""
        factors = []
        
        if not dog_stats:
            factors.append("No historical data available")
            return factors
        
        # Performance factors
        if dog_stats.get('win_rate', 0) > 0.3:
            factors.append(f"Strong win rate ({dog_stats['win_rate']:.1%})")
        
        if dog_stats.get('form_trend', 0) > 0.5:
            factors.append("Improving form")
        elif dog_stats.get('form_trend', 0) < -0.5:
            factors.append("Declining form")
        
        # Consistency
        if dog_stats.get('position_consistency', 0) > 0.8:
            factors.append("Highly consistent")
        
        # Recent activity
        activity = dog_stats.get('recent_activity', {})
        if isinstance(activity, dict):
            days_since = activity.get('days_since_last_race', 30)
            if days_since < 14:
                factors.append("Recently active")
            elif days_since > 60:
                factors.append("Long layoff")
        
        # Experience
        races_count = dog_stats.get('races_count', 0)
        if races_count < 3:
            factors.append("Limited experience")
        elif races_count > 15:
            factors.append("Highly experienced")
        
        # Context factors
        if race_context:
            # Track condition preference
            track_perf = dog_stats.get('track_condition_performance', {})
            current_condition = race_context.get('track_condition', 'Good')
            if current_condition in track_perf and track_perf[current_condition]['races'] >= 2:
                win_rate = track_perf[current_condition]['win_rate']
                if win_rate > 0.25:
                    factors.append(f"Good on {current_condition.lower()} tracks")
        
        return factors
    
    def train_ml_models(self):
        """Train enhanced machine learning models with more data and features"""
        if not SKLEARN_AVAILABLE:
            print("⚠️ Scikit-learn not available for ML training")
            return
        
        # First check if we have sufficient database records
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NOT NULL")
        total_records = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT dog_clean_name) FROM dog_race_data WHERE finish_position IS NOT NULL")
        unique_dogs = cursor.fetchone()[0]
        conn.close()
        
        if total_records < 500 or unique_dogs < 50:
            print(f"⚠️ Insufficient database records for ML training ({total_records} records, {unique_dogs} dogs)")
            print("   Using traditional scoring methods with CSV form data instead")
            return
        
        print(f"🤖 Training enhanced ML models with {total_records} records from {unique_dogs} dogs...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load more comprehensive training data
            query = """
            SELECT 
                drd.dog_clean_name,
                drd.finish_position,
                drd.box_number,
                drd.weight,
                drd.starting_price,
                drd.performance_rating,
                drd.speed_rating,
                drd.class_rating,
                drd.individual_time,
                drd.margin,
                rm.field_size,
                rm.distance,
                rm.venue,
                rm.track_condition,
                rm.weather,
                rm.temperature,
                rm.grade,
                rm.race_date
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
            AND drd.finish_position != ''
            AND drd.finish_position != 'N/A'
            AND drd.individual_time IS NOT NULL
            ORDER BY rm.race_date DESC
            LIMIT 5000
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Get historical performance for each dog to create enhanced features
            print("   📊 Creating enhanced features from historical data...")
            enhanced_features = []
            
            for _, row in df.iterrows():
                dog_name = row['dog_clean_name']
                race_date = row['race_date']
                
                # Get historical performance up to this race date
                hist_query = """
                SELECT finish_position, individual_time, margin, box_number, starting_price
                FROM dog_race_data drd2
                JOIN race_metadata rm2 ON drd2.race_id = rm2.race_id
                WHERE drd2.dog_clean_name = ?
                AND rm2.race_date < ?
                AND drd2.finish_position IS NOT NULL
                ORDER BY rm2.race_date DESC
                LIMIT 10
                """
                
                cursor = conn.cursor()
                cursor.execute(hist_query, (dog_name, race_date))
                hist_results = cursor.fetchall()
                
                if len(hist_results) >= 1:  # Minimum historical data (lowered to use available data)
                    # Calculate enhanced features
                    hist_positions = [int(r[0]) for r in hist_results if r[0] and str(r[0]).isdigit()]
                    hist_times = [float(r[1]) for r in hist_results if r[1]]
                    hist_margins = [float(r[2]) for r in hist_results if r[2]]
                    hist_boxes = [int(r[3]) for r in hist_results if r[3] and str(r[3]).isdigit()]
                    hist_odds = [float(r[4]) for r in hist_results if r[4]]
                    
                    # Enhanced feature calculations
                    recent_form = np.mean(hist_positions[:3]) if len(hist_positions) >= 3 else 5
                    form_trend = -np.polyfit(range(len(hist_positions)), hist_positions, 1)[0] if len(hist_positions) >= 3 else 0
                    consistency = 1 / (np.std(hist_positions) + 1) if len(hist_positions) > 1 else 0.5
                    avg_margin = np.mean([abs(m) for m in hist_margins]) if hist_margins else 2.0
                    time_consistency = 1 / (np.std(hist_times) + 0.1) if len(hist_times) > 1 else 0.5
                    box_versatility = len(set(hist_boxes)) if hist_boxes else 1
                    market_confidence = 1 / (np.mean(hist_odds) + 1) if hist_odds else 0.1
                    
                    feature_row = {
                        'finish_position': row['finish_position'],
                        'box_number': row['box_number'],
                        'weight': row['weight'],
                        'starting_price': row['starting_price'],
                        'performance_rating': row['performance_rating'],
                        'speed_rating': row['speed_rating'],
                        'class_rating': row['class_rating'],
                        'field_size': row['field_size'],
                        'distance': row['distance'],
                        'venue': row['venue'],
                        'track_condition': row['track_condition'],
                        # Enhanced features
                        'recent_form': recent_form,
                        'form_trend': form_trend,
                        'consistency': consistency,
                        'avg_margin': avg_margin,
                        'time_consistency': time_consistency,
                        'box_versatility': box_versatility,
                        'market_confidence': market_confidence,
                        'historical_races': len(hist_results)
                    }
                    enhanced_features.append(feature_row)
            
            conn.close()
            
            if len(enhanced_features) < 100:
                print("⚠️ Insufficient enhanced data for ML training")
                return
            
            # Convert to DataFrame and prepare
            enhanced_df = pd.DataFrame(enhanced_features)
            enhanced_df = self.prepare_enhanced_ml_dataframe(enhanced_df)
            
            if enhanced_df is None or len(enhanced_df) < 100:
                print("⚠️ Could not prepare enhanced ML features")
                return
            
            # Create multiple target variables
            enhanced_df['is_winner'] = (enhanced_df['finish_position'] == 1).astype(int)
            enhanced_df['is_placer'] = (enhanced_df['finish_position'] <= 3).astype(int)
            enhanced_df['top_half'] = (enhanced_df['finish_position'] <= 4).astype(int)
            
            # Enhanced feature selection
            feature_columns = [
                'box_number', 'weight', 'odds_log', 'performance_rating',
                'speed_rating', 'class_rating', 'field_size', 'distance_numeric',
                'venue_encoded', 'track_condition_encoded',
                'recent_form', 'form_trend', 'consistency', 'avg_margin',
                'time_consistency', 'box_versatility', 'market_confidence', 'historical_races'
            ]
            
            # Filter to complete cases
            complete_df = enhanced_df[feature_columns + ['is_winner', 'is_placer']].dropna()
            
            if len(complete_df) < 100:
                print("⚠️ Insufficient complete enhanced data for ML training")
                return
            
            X = complete_df[feature_columns]
            y_win = complete_df['is_winner']
            y_place = complete_df['is_placer']
            
            # Train-test split
            X_train, X_test, y_win_train, y_win_test, y_place_train, y_place_test = train_test_split(
                X, y_win, y_place, test_size=0.2, random_state=42, stratify=y_win
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble models
            # Win prediction model
            rf_win = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            gb_win = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr_win = LogisticRegression(random_state=42, max_iter=1000)
            
            win_ensemble = VotingClassifier([
                ('rf', rf_win),
                ('gb', gb_win),
                ('lr', lr_win)
            ], voting='soft')
            
            win_ensemble.fit(X_train_scaled, y_win_train)
            
            # Place prediction model
            rf_place = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            gb_place = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            place_ensemble = VotingClassifier([
                ('rf', rf_place),
                ('gb', gb_place)
            ], voting='soft')
            
            place_ensemble.fit(X_train_scaled, y_place_train)
            
            # Evaluate models
            win_accuracy = win_ensemble.score(X_test_scaled, y_win_test)
            place_accuracy = place_ensemble.score(X_test_scaled, y_place_test)
            
            # Store models and scalers
            self.ml_models['race_winner'] = win_ensemble
            self.ml_models['race_placer'] = place_ensemble
            self.scalers['race_winner'] = scaler
            self.scalers['race_placer'] = scaler
            
            # Store feature importance (from random forest component within ensemble)
            try:
                # Access the fitted random forest from the ensemble
                fitted_rf = win_ensemble.named_estimators_['rf']
                self.feature_importance = dict(zip(feature_columns, fitted_rf.feature_importances_))
            except:
                # Fallback if feature importance extraction fails
                self.feature_importance = {col: 1.0/len(feature_columns) for col in feature_columns}
            
            print(f"✅ Enhanced ML models trained")
            print(f"   🏆 Win prediction accuracy: {win_accuracy:.3f}")
            print(f"   🥉 Place prediction accuracy: {place_accuracy:.3f}")
            print(f"   📊 Training samples: {len(complete_df)}")
            print(f"   🔧 Features used: {len(feature_columns)}")
            print(f"   📈 Top features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
        except Exception as e:
            print(f"❌ Enhanced ML training error: {e}")
    
    def prepare_enhanced_ml_dataframe(self, df):
        """Prepare enhanced dataframe for ML training with more features"""
        try:
            # Clean and convert data types
            df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
            df['box_number'] = pd.to_numeric(df['box_number'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['starting_price'] = pd.to_numeric(df['starting_price'], errors='coerce')
            df['field_size'] = pd.to_numeric(df['field_size'], errors='coerce')
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
            
            # Create engineered features
            df['odds_log'] = np.log(df['starting_price'].fillna(10) + 1)
            df['distance_numeric'] = df['distance'].fillna(500)
            
            # Encode categorical variables with more sophisticated handling
            venues = df['venue'].fillna('Unknown')
            conditions = df['track_condition'].fillna('Good')
            
            if not hasattr(self, 'venue_encoder'):
                self.venue_encoder = LabelEncoder()
                self.condition_encoder = LabelEncoder()
                
                df['venue_encoded'] = self.venue_encoder.fit_transform(venues)
                df['track_condition_encoded'] = self.condition_encoder.fit_transform(conditions)
            else:
                df['venue_encoded'] = self.encode_with_unknown(venues, self.venue_encoder)
                df['track_condition_encoded'] = self.encode_with_unknown(conditions, self.condition_encoder)
            
            # Fill missing values with more intelligent defaults
            numeric_columns = ['performance_rating', 'speed_rating', 'class_rating']
            for col in numeric_columns:
                df[col] = df[col].fillna(50)  # Default rating
            
            # Enhanced feature defaults
            df['recent_form'] = df['recent_form'].fillna(5.0)
            df['form_trend'] = df['form_trend'].fillna(0.0)
            df['consistency'] = df['consistency'].fillna(0.5)
            df['avg_margin'] = df['avg_margin'].fillna(2.0)
            df['time_consistency'] = df['time_consistency'].fillna(0.5)
            df['box_versatility'] = df['box_versatility'].fillna(1)
            df['market_confidence'] = df['market_confidence'].fillna(0.1)
            df['historical_races'] = df['historical_races'].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"❌ Enhanced ML dataframe preparation error: {e}")
            return None
    
    def prepare_ml_dataframe(self, df):
        """Prepare dataframe for ML training"""
        try:
            # Clean and convert data types
            df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
            df['box_number'] = pd.to_numeric(df['box_number'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['starting_price'] = pd.to_numeric(df['starting_price'], errors='coerce')
            df['field_size'] = pd.to_numeric(df['field_size'], errors='coerce')
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
            df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
            
            # Create engineered features
            df['odds_log'] = np.log(df['starting_price'].fillna(10) + 1)
            df['distance_numeric'] = df['distance'].fillna(500)
            
            # Encode categorical variables
            venues = df['venue'].fillna('Unknown')
            conditions = df['track_condition'].fillna('Good')
            
            if not hasattr(self, 'venue_encoder'):
                self.venue_encoder = LabelEncoder()
                self.condition_encoder = LabelEncoder()
                
                df['venue_encoded'] = self.venue_encoder.fit_transform(venues)
                df['track_condition_encoded'] = self.condition_encoder.fit_transform(conditions)
            else:
                # Handle new unseen categories
                df['venue_encoded'] = self.encode_with_unknown(venues, self.venue_encoder)
                df['track_condition_encoded'] = self.encode_with_unknown(conditions, self.condition_encoder)
            
            # Fill missing values
            numeric_columns = ['performance_rating', 'speed_rating', 'class_rating']
            for col in numeric_columns:
                df[col] = df[col].fillna(50)  # Default rating
            
            return df
            
        except Exception as e:
            print(f"❌ ML dataframe preparation error: {e}")
            return None
    
    def encode_with_unknown(self, data, encoder):
        """Encode data handling unknown categories"""
        try:
            encoded = []
            for item in data:
                try:
                    encoded.append(encoder.transform([item])[0])
                except ValueError:
                    # Unknown category - use most common class
                    encoded.append(0)
            return encoded
        except:
            return [0] * len(data)
    
    def prepare_ml_features(self, dog_stats, race_context):
        """Prepare features for ML prediction with enhanced features matching training"""
        try:
            if not dog_stats:
                return None
            
            # Extract features matching the 18 features used in training
            features = [
                race_context.get('field_size', 8),  # box_number - use field_size as proxy
                dog_stats.get('avg_weight', 30.0),  # weight
                np.log(dog_stats.get('avg_starting_price', 10.0) + 1),  # odds_log
                dog_stats.get('avg_performance_rating', 50),  # performance_rating
                dog_stats.get('avg_speed_rating', 50),  # speed_rating
                dog_stats.get('avg_class_rating', 50),  # class_rating
                race_context.get('field_size', 8),  # field_size
                float(race_context.get('distance', '500')),  # distance_numeric
                0,  # venue_encoded (would need proper encoding)
                0,  # track_condition_encoded (would need proper encoding)
                # Enhanced features (matching training features)
                dog_stats.get('avg_position', 5.0),  # recent_form
                dog_stats.get('form_trend', 0.0),  # form_trend
                dog_stats.get('position_consistency', 0.5),  # consistency
                dog_stats.get('avg_margin', 2.0),  # avg_margin
                dog_stats.get('time_consistency', 0.5),  # time_consistency
                dog_stats.get('box_versatility', 1),  # box_versatility
                dog_stats.get('market_confidence', 0.1),  # market_confidence
                dog_stats.get('races_count', 0)  # historical_races
            ]
            
            return features
            
        except Exception as e:
            print(f"Warning: ML feature preparation error: {e}")
            return None
    
    def get_current_weather(self, venue):
        """Get current weather data for venue"""
        if not self.weather_api_key or not REQUESTS_AVAILABLE:
            return None
        
        # Venue coordinates (approximate)
        venue_coords = {
            'AP_K': (-34.7, 138.5),  # Angle Park
            'DAPT': (-34.4, 150.9),  # Dapto
            'CASO': (-28.8, 153.3),  # Casino
            'MEA': (-37.7, 144.7),   # The Meadows
            'SAN': (-37.9, 145.1),  # Sandown
            'HOBT': (-42.9, 147.3),  # Hobart
            'GAWL': (-34.6, 138.7),  # Gawler
        }
        
        coords = venue_coords.get(venue)
        if not coords:
            return None
        
        try:
            lat, lon = coords
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'weather': data['weather'][0]['description'],
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'track_condition': self.weather_to_track_condition(data)
                }
        except Exception as e:
            print(f"⚠️ Weather API error: {e}")
        
        return None
    
    def weather_to_track_condition(self, weather_data):
        """Convert weather data to likely track condition"""
        try:
            description = weather_data['weather'][0]['description'].lower()
            
            if any(word in description for word in ['rain', 'drizzle', 'shower']):
                return 'Slow'
            elif any(word in description for word in ['heavy rain', 'thunderstorm']):
                return 'Heavy'
            elif 'cloud' in description:
                return 'Good'
            else:
                return 'Fast'
        except:
            return 'Good'
    
    def get_market_odds(self, race_info, dogs):
        """Get current market odds (placeholder for real bookmaker API integration)"""
        # This would integrate with real bookmaker APIs like:
        # - Bet365 API
        # - William Hill API  
        # - Ladbrokes API
        # - TAB API (Australia specific)
        
        # Return empty dict - no fake odds generation
        # Real implementation would fetch actual market data
        return {}
    
    def calculate_value_bets(self, predictions, market_odds):
        """Calculate value betting opportunities"""
        value_bets = []
        
        for prediction in predictions:
            dog_name = prediction.get('clean_name', prediction.get('dog_name', ''))
            our_probability = prediction.get('prediction_score', 0)
            
            if dog_name in market_odds:
                market_data = market_odds[dog_name]
                market_odds_decimal = market_data.get('win_odds', 10.0)
                
                # Convert odds to implied probability
                market_probability = 1 / market_odds_decimal if market_odds_decimal > 0 else 0.1
                
                # Calculate value (our probability vs market probability)
                if our_probability > market_probability * 1.1:  # 10% edge threshold
                    value_percentage = ((our_probability / market_probability) - 1) * 100
                    
                    value_bet = {
                        'dog_name': dog_name,
                        'our_probability': our_probability,
                        'market_probability': market_probability,
                        'market_odds': market_odds_decimal,
                        'value_percentage': value_percentage,
                        'confidence': prediction.get('confidence_level', 0.5),
                        'recommendation': self.get_value_bet_recommendation(value_percentage, our_probability)
                    }
                    
                    value_bets.append(value_bet)
        
        # Sort by value percentage
        value_bets.sort(key=lambda x: x['value_percentage'], reverse=True)
        return value_bets
    
    def get_value_bet_recommendation(self, value_percentage, probability):
        """Get value betting recommendation"""
        if value_percentage > 50 and probability > 0.4:
            return "Strong Value Bet"
        elif value_percentage > 25 and probability > 0.3:
            return "Good Value"
        elif value_percentage > 10 and probability > 0.2:
            return "Small Value"
        else:
            return "Monitor"
    
    def analyze_market_sentiment(self, predictions, market_odds):
        """Analyze market sentiment vs our predictions"""
        sentiment_analysis = {
            'market_vs_prediction': [],
            'overvalued_dogs': [],
            'undervalued_dogs': [],
            'market_efficiency': 0
        }
        
        total_variance = 0
        comparison_count = 0
        
        for prediction in predictions:
            dog_name = prediction.get('clean_name', prediction.get('dog_name', ''))
            our_prob = prediction.get('prediction_score', 0)
            
            if dog_name in market_odds:
                market_odds_decimal = market_odds[dog_name].get('win_odds', 10.0)
                market_prob = 1 / market_odds_decimal if market_odds_decimal > 0 else 0.1
                
                variance = abs(our_prob - market_prob)
                total_variance += variance
                comparison_count += 1
                
                comparison = {
                    'dog_name': dog_name,
                    'our_probability': our_prob,
                    'market_probability': market_prob,
                    'difference': our_prob - market_prob,
                    'variance': variance
                }
                
                sentiment_analysis['market_vs_prediction'].append(comparison)
                
                # Identify overvalued/undervalued
                if our_prob > market_prob * 1.2:  # We rate 20% higher
                    sentiment_analysis['undervalued_dogs'].append(comparison)
                elif market_prob > our_prob * 1.2:  # Market rates 20% higher
                    sentiment_analysis['overvalued_dogs'].append(comparison)
        
        # Calculate market efficiency (lower variance = more efficient)
        if comparison_count > 0:
            avg_variance = total_variance / comparison_count
            sentiment_analysis['market_efficiency'] = max(0, min(1, 1 - (avg_variance * 2)))
        
        return sentiment_analysis
    
    def check_and_update_completed_races(self):
        """Check for completed races and update database with results"""
        print("🔄 Checking for completed races to update database...")
        
        if not self.upcoming_dir.exists():
            return
        
        csv_files = list(self.upcoming_dir.glob('*.csv'))
        updated_count = 0
        
        for csv_file in csv_files:
            try:
                # Check if this race has results available
                if self.has_race_completed(csv_file):
                    if self.update_database_with_results(csv_file):
                        updated_count += 1
                        print(f"   ✅ Updated database with results from {csv_file.name}")
                        
                        # Move completed race file to processed folder
                        processed_path = Path('./processed_races') / csv_file.name
                        processed_path.parent.mkdir(exist_ok=True)
                        csv_file.rename(processed_path)
                        
            except Exception as e:
                print(f"   ⚠️ Error processing {csv_file.name}: {e}")
        
        if updated_count > 0:
            print(f"📊 Updated database with {updated_count} completed races")
        else:
            print("📊 No new completed races found")
    
    def has_race_completed(self, csv_file):
        """Check if a race has completed by looking for results data"""
        try:
            df = pd.read_csv(csv_file)
            
            # Check if any dog has a finish position recorded
            if 'PLC' in df.columns:
                has_results = df['PLC'].notna().any() and (df['PLC'] != '').any()
                return has_results
            
            return False
        except Exception:
            return False
    
    def update_database_with_results(self, csv_file):
        """Update database with race results from completed race CSV using enhanced processor"""
        try:
            # Use enhanced race processor if available for better dead heat handling
            if self.use_enhanced_processor and self.race_processor:
                print(f"   🚀 Using enhanced race processor for {csv_file.name}")
                result = self.race_processor.process_race_results(csv_file, move_processed=True)
                
                if result.get('success') or result.get('status') == 'success_with_issues':
                    print(f"   ✅ Enhanced processing successful: {result.get('summary', '')}")
                    if result.get('moved_to'):
                        print(f"   📁 File moved to: {result['moved_to']}")
                    return True
                else:
                    print(f"   ⚠️ Enhanced processing failed, falling back to legacy method")
                    return self._legacy_update_database_with_results(csv_file)
            else:
                # Fall back to legacy method
                return self._legacy_update_database_with_results(csv_file)
            
        except Exception as e:
            print(f"Error updating database with results from {csv_file}: {e}")
            return False
    
    def _legacy_update_database_with_results(self, csv_file):
        """Legacy database update method with basic dead heat handling"""
        try:
            df = pd.read_csv(csv_file)
            filename = csv_file.name
            race_info = self.extract_race_info(filename)
            
            # Generate race ID
            race_id = f"{race_info.get('venue', 'UNK')}_{race_info.get('race_number', '0')}_{race_info.get('date_str', 'UNKNOWN').replace(' ', '_')}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert/update race metadata
            cursor.execute("""
                INSERT OR REPLACE INTO race_metadata (
                    race_id, venue, race_number, race_date, distance, field_size,
                    extraction_timestamp, data_source, race_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                race_info.get('venue', 'Unknown'),
                race_info.get('race_number', 0),
                race_info.get('date_str', datetime.now().strftime('%Y-%m-%d')),
                df.get('DIST', {}).iloc[0] if 'DIST' in df.columns and not df.empty else '500',
                len(df),
                datetime.now().isoformat(),
                'legacy_completed_race_update',
                'completed'
            ))
            
            # Handle possible dead heats
            def determine_finish_positions(results):
                positions = {}
                for i, dog_name in enumerate(results, start=1):
                    if dog_name not in positions:
                        positions[dog_name] = str(i)
                    else:
                        positions[dog_name] += "="  # Mark as dead heat
                return positions
            
            # Determine finish positions with dead heat handling
            finish_positions = []
            for column in ['first', 'second', 'third', 'fourth']:
                if column in df.columns:
                    finish_positions.append(df[column].fillna('').tolist())
            
            # Flatten and map finish positions
            flattened_positions = [item for sublist in finish_positions for item in sublist]
            position_dict = determine_finish_positions(flattened_positions)
            
            # Process each dog's results with determined positions
            for _, row in df.iterrows():
                raw_dog_name = str(row['Dog Name']).strip()
                if pd.isna(raw_dog_name) or not raw_dog_name:
                    continue

                clean_name = re.sub(r'^["\d\.\s]+', '', raw_dog_name).strip().upper()
                finish_position = position_dict.get(clean_name, None)
                
                # Insert/update dog race data
                cursor.execute("""
                    INSERT OR REPLACE INTO dog_race_data (
                        race_id, dog_name, dog_clean_name, box_number, finish_position,
                        trainer_name, weight, individual_time, margin, starting_price,
                        extraction_timestamp, data_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    raw_dog_name,
                    clean_name,
                    row.get('BOX', None),
                    finish_position,
                    row.get('TRAINER', None),
                    row.get('WGT', None),
                    row.get('TIME', None),
                    row.get('MGN', None),
                    row.get('ODDS', None),
                    datetime.now().isoformat(),
                    'legacy_completed_race_update'
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error in legacy database update from {csv_file}: {e}")
            return False
    
    def predict_race(self, race_file_path):
        """Predict outcomes for a single race using enhanced analysis"""
        try:
            df = pd.read_csv(race_file_path)
            
            # Extract race information
            filename = os.path.basename(race_file_path)
            race_info = self.extract_race_info(filename)
            
            # Get current race context (weather, track conditions, etc.)
            race_context = self.get_race_context(race_info, df)
            
            predictions = []
            
            # Count actual dogs (rows with numbered names)
            dog_count = 0
            for _, row in df.iterrows():
                raw_dog_name = str(row.get('Dog Name', row.get('DOG', ''))).strip()
                if raw_dog_name and raw_dog_name.lower() != 'nan' and raw_dog_name != '""':
                    if any(char.isdigit() for char in raw_dog_name[:3]):  # Has number prefix
                        dog_count += 1
            
            print(f"   🎯 Analyzing {dog_count} dogs with enhanced metrics...")
            
            # Train ML models once if we have dogs with data
            ml_trained = False
            if self.use_ml_predictions:
                print("🤖 Training enhanced ML models with more data...")
                self.train_ml_models()
                ml_trained = True
            
            for _, row in df.iterrows():
                # Get dog name - try multiple possible column names
                raw_dog_name = str(row.get('Dog Name', row.get('DOG', ''))).strip()
                if not raw_dog_name or raw_dog_name.lower() == 'nan' or raw_dog_name == '""':
                    continue
                
                import re
                box_number_pattern = r'^(\d+)\.'  # Match number before the period
                match_box = re.match(box_number_pattern, raw_dog_name)
                box_number = match_box.group(1) if match_box else ''

                # Extract clean dog name from numbered format
                dog_name = re.sub(box_number_pattern, '', raw_dog_name).strip()
                # Clean dog name with fuzzy matching if available  
                clean_name = self.enhanced_clean_dog_name(dog_name)
                
                # Get comprehensive historical performance, passing df for form data extraction
                dog_stats = self.get_comprehensive_dog_performance(clean_name, df)
                
                # Enhanced prediction score with context
                prediction_score = self.calculate_enhanced_prediction_score(dog_stats, race_context)
                
                # ML prediction if available (models already trained)
                ml_prediction_score = None
                if ml_trained and dog_stats and 'race_winner' in self.ml_models:
                    ml_prediction_score = self.get_ml_prediction(dog_stats, race_context)
                
                # Combine traditional and ML scores
                final_score = self.combine_prediction_scores(prediction_score, ml_prediction_score)
                
                # Get other race info
                # Assign box number from parsed name if available; otherwise, use CSV
                if not box_number:
                    box_number = row.get('BOX', '')
                trainer = row.get('TRAINER', '')
                weight = row.get('WEIGHT', '')
                
                # Enhanced prediction object
                prediction = {
                    'dog_name': dog_name,
                    'clean_name': clean_name,
                    'box_number': box_number,
                    'trainer': trainer,
                    'weight': weight,
                    'prediction_score': final_score,
                    'traditional_score': prediction_score,
                    'ml_score': ml_prediction_score,
                    'historical_stats': dog_stats,
                    'recommended_bet': self.get_enhanced_bet_recommendation(final_score, dog_stats),
                    'confidence_level': self.calculate_confidence_level(dog_stats),
                    'key_factors': self.identify_key_factors(dog_stats, race_context)
                }
                
                predictions.append(prediction)
                print(f"     📊 {dog_name}: {final_score:.1%} confidence")
            
            # Sort by prediction score (highest first)
            predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
            
            # Add rankings
            for i, pred in enumerate(predictions):
                pred['predicted_rank'] = i + 1
            
            # Get market odds and value analysis
            market_odds = self.get_market_odds(race_info, predictions)
            value_bets = self.calculate_value_bets(predictions, market_odds)
            market_sentiment = self.analyze_market_sentiment(predictions, market_odds)
            
            # Enhanced race summary
            race_summary = {
                'total_dogs': len(predictions),
                'dogs_with_data': len([p for p in predictions if p.get('historical_stats')]),
                'average_confidence': np.mean([p.get('confidence_level', 0) for p in predictions]),
                'ml_predictions_used': any(p.get('ml_score') is not None for p in predictions),
                'market_analysis': market_sentiment,
                'value_opportunities': len(value_bets)
            }
            
            return {
                'race_info': race_info,
                'race_context': race_context,
                'predictions': predictions,
                'top_pick': predictions[0] if predictions else None,
                'market_odds': market_odds,
                'value_bets': value_bets,
                'market_sentiment': market_sentiment,
                'race_summary': race_summary,
                'prediction_timestamp': datetime.now().isoformat(),
                'analysis_version': '2.0_enhanced'
            }
            
        except Exception as e:
            print(f"Error predicting race {race_file_path}: {e}")
            return None
    
    def extract_race_info(self, filename):
        """Extract race information from filename"""
        import re
        
        # Extract race number, venue, and date
        race_pattern = r'Race\s+(\d+)\s+-\s+([A-Z]+)\s+-\s+(\d{1,2}\s+\w+\s+\d{4})'
        match = re.search(race_pattern, filename, re.IGNORECASE)
        
        if match:
            race_number, venue, date_str = match.groups()
            return {
                'race_number': int(race_number),
                'venue': venue,
                'date_str': date_str,
                'filename': filename
            }
        
        return {'filename': filename}
    
    def clean_dog_name(self, name):
        """Clean dog name for database lookup"""
        if not name:
            return ""
        
        # Remove quotes, numbers, and extra spaces
        import re
        cleaned = re.sub(r'^["\d\.\s]+', '', str(name))
        cleaned = re.sub(r'["\s]+$', '', cleaned)
        return cleaned.strip().upper()
    
    def get_bet_recommendation(self, score):
        """Get betting recommendation based on prediction score"""
        if score >= 0.7:
            return "Strong Win"
        elif score >= 0.5:
            return "Win/Place"
        elif score >= 0.3:
            return "Place Only"
        else:
            return "Avoid"
    
    def predict_all_upcoming_races(self):
        """Predict all upcoming races"""
        print("🎯 PREDICTING UPCOMING RACES")
        print("=" * 50)
        
        # First, check for and update completed races
        self.check_and_update_completed_races()
        
        if not self.upcoming_dir.exists():
            print(f"⚠️  Upcoming races directory not found: {self.upcoming_dir}")
            return
        
        csv_files = list(self.upcoming_dir.glob('*.csv'))
        
        # Filter out README.md and other non-race files
        csv_files = [f for f in csv_files if f.name != 'README.md']
        
        if not csv_files:
            print("ℹ️  No upcoming race files found")
            return
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        print(f"🔍 Found {len(csv_files)} upcoming races to predict")
        print(f"📅 Processing in order of upload (newest first):")
        for i, f in enumerate(csv_files, 1):
            import time
            mod_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))
            print(f"   {i}. {f.name} (uploaded: {mod_time})")
        
        all_predictions = []
        
        for file_path in csv_files:
            print(f"\n🎯 Predicting: {file_path.name}")
            
            prediction = self.predict_race(file_path)
            
            if prediction:
                all_predictions.append(prediction)
                
                # Save individual prediction
                pred_filename = f"prediction_{file_path.stem}.json"
                pred_path = self.predictions_dir / pred_filename
                
                with open(pred_path, 'w') as f:
                    json.dump(prediction, f, indent=2, default=str)
                
                print(f"   ✅ Saved prediction to {pred_filename}")
                
                # Show enhanced analysis results
                if prediction['predictions']:
                    print(f"   🏆 Top 3 picks:")
                    for i, pred in enumerate(prediction['predictions'][:3]):
                        confidence = pred.get('confidence_level', 0)
                        key_factors = pred.get('key_factors', [])
                        factors_str = f" ({', '.join(key_factors[:2])})" if key_factors else ""
                        print(f"      {i+1}. {pred['dog_name']} (Box {pred['box_number']}) - {pred['prediction_score']:.1%} - {pred['recommended_bet']} [Confidence: {confidence:.1%}]{factors_str}")
                    
                    # Show market analysis
                    value_bets = prediction.get('value_bets', [])
                    if value_bets:
                        print(f"   💰 Value betting opportunities:")
                        for vb in value_bets[:2]:  # Show top 2 value bets
                            print(f"      🎯 {vb['dog_name']}: {vb['value_percentage']:.1f}% value - {vb['recommendation']}")
                    
                    # Show race summary
                    summary = prediction.get('race_summary', {})
                    dogs_with_data = summary.get('dogs_with_data', 0)
                    total_dogs = summary.get('total_dogs', len(prediction['predictions']))
                    print(f"   📊 Analysis: {dogs_with_data}/{total_dogs} dogs with historical data")
                    
                    if summary.get('ml_predictions_used'):
                        print(f"   🤖 ML predictions included")
            else:
                print(f"   ❌ Failed to predict race")
        
        # Save summary
        summary = {
            'total_races': len(csv_files),
            'successful_predictions': len(all_predictions),
            'predictions': all_predictions,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.predictions_dir / f"prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 PREDICTION SUMMARY")
        print(f"   Total races: {len(csv_files)}")
        print(f"   Successful predictions: {len(all_predictions)}")
        print(f"   Summary saved to: {summary_path.name}")
        print(f"   ✅ Prediction complete!")

def main():
    """Main function"""
    import sys
    
    predictor = UpcomingRacePredictor()
    
    # Check if a specific race file was provided as argument
    if len(sys.argv) > 1:
        race_file_path = sys.argv[1]
        print(f"🎯 Predicting single race: {race_file_path}")
        
        # Predict single race
        prediction = predictor.predict_race(race_file_path)
        
        if prediction:
            # Save individual prediction
            import os
            filename = os.path.basename(race_file_path)
            pred_filename = f"prediction_{filename.replace('.csv', '')}.json"
            pred_path = predictor.predictions_dir / pred_filename
            
            with open(pred_path, 'w') as f:
                json.dump(prediction, f, indent=2, default=str)
            
            print(f"   ✅ Saved prediction to {pred_filename}")
            
            # Show enhanced analysis results
            if prediction['predictions']:
                print(f"   🏆 Top 3 picks:")
                for i, pred in enumerate(prediction['predictions'][:3]):
                    confidence = pred.get('confidence_level', 0)
                    key_factors = pred.get('key_factors', [])
                    factors_str = f" ({', '.join(key_factors[:2])})" if key_factors else ""
                    print(f"      {i+1}. {pred['dog_name']} (Box {pred['box_number']}) - {pred['prediction_score']:.1%} - {pred['recommended_bet']} [Confidence: {confidence:.1%}]{factors_str}")
                
                # Show race summary
                summary = prediction.get('race_summary', {})
                dogs_with_data = summary.get('dogs_with_data', 0)
                total_dogs = summary.get('total_dogs', len(prediction['predictions']))
                print(f"   📊 Analysis: {dogs_with_data}/{total_dogs} dogs with historical data")
                
                if summary.get('ml_predictions_used'):
                    print(f"   🤖 ML predictions included")
        else:
            print(f"   ❌ Failed to predict race")
    else:
        # Default behavior - predict all upcoming races
        predictor.predict_all_upcoming_races()

if __name__ == "__main__":
    main()
def predict_upcoming_race(csv_file_path):
    predictor = UpcomingRacePredictor()
    result = predictor.predict_race(csv_file_path)
    if result and 'predictions' in result:
        # Return just dog names and scores for the /predict page
        return [f"{p['dog_name']} - {p['prediction_score']:.0%} chance to win ({p['recommended_bet']})"
                for p in result['predictions']]
    else:
        return ["No prediction available"]

