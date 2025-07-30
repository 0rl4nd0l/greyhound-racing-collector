#!/usr/bin/env python3
"""
Fixed Prediction Scoring System
===============================

This module fixes the critical issues causing low prediction scores:
1. Removes random noise injection
2. Improves historical data aggregation from race files
3. Enhances feature engineering algorithms
4. Adds better fallback logic

Author: AI Assistant
Date: July 27, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FixedPredictionScoring:
    """Enhanced prediction scoring with proper feature extraction"""
    
    def __init__(self):
        self.min_races_for_reliable_prediction = 3
        
    def calculate_comprehensive_historical_stats(self, dog_data, race_info):
        """Calculate meaningful historical statistics from all available data"""
        stats = {
            'total_races': 0,
            'win_rate': 0.0,
            'place_rate': 0.0,
            'average_position': 8.0,  # Worst case default
            'consistency_score': 0.0,
            'recent_form_trend': 0.0,
            'venue_experience': 0,
            'venue_win_rate': 0.0,
            'best_time': None,
            'average_time': None,
            'weight_consistency': 0.0,
            'box_preference': 0.0
        }
        
        # Combine all historical data sources
        all_races = []
        
        # Process race file embedded data (multi-row format)
        if dog_data.get('form_guide_data'):
            for race in dog_data['form_guide_data']:
                try:
                    processed_race = {
                        'place': self._safe_int(race.get('place', 8)),
                        'time': self._safe_float(race.get('time', 0)),
                        'weight': self._safe_float(race.get('weight', 0)),
                        'box': self._safe_int(race.get('box', 1)),
                        'venue': race.get('track', ''),
                        'date': race.get('date', ''),
                        'distance': self._safe_int(race.get('distance', 0))
                    }
                    all_races.append(processed_race)
                except Exception as e:
                    continue
        
        # Process database data
        if dog_data.get('database_data'):
            for race in dog_data['database_data']:
                try:
                    processed_race = {
                        'place': self._safe_int(race.get('finish_position', 8)),
                        'time': self._safe_float(race.get('individual_time', 0)),
                        'weight': self._safe_float(race.get('weight', 0)),
                        'box': self._safe_int(race.get('box_number', 1)),
                        'venue': race.get('venue', ''),
                        'date': race.get('race_date', ''),
                        'distance': self._safe_int(race.get('distance', 0))
                    }
                    all_races.append(processed_race)
                except Exception as e:
                    continue
        
        if not all_races:
            return stats
        
        # Sort by date (newest first) for recent form analysis
        all_races.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate basic statistics
        stats['total_races'] = len(all_races)
        
        positions = [race['place'] for race in all_races if race['place'] > 0]
        if positions:
            stats['win_rate'] = sum(1 for p in positions if p == 1) / len(positions)
            stats['place_rate'] = sum(1 for p in positions if p <= 3) / len(positions)
            stats['average_position'] = np.mean(positions)
            
            # Calculate consistency (lower standard deviation = more consistent)
            stats['consistency_score'] = max(0, 1.0 - (np.std(positions) / 4.0))
        
        # Recent form trend (last 5 vs previous 5 races)
        if len(positions) >= 6:
            recent_5 = positions[:5]
            previous_5 = positions[5:10] if len(positions) >= 10 else positions[5:]
            
            if previous_5:
                recent_avg = np.mean(recent_5)
                previous_avg = np.mean(previous_5)
                # Positive trend means improving (lower average position)
                stats['recent_form_trend'] = (previous_avg - recent_avg) / 4.0
        
        # Venue-specific analysis
        current_venue = race_info.get('venue', '')
        venue_races = [r for r in all_races if r['venue'] == current_venue]
        if venue_races:
            stats['venue_experience'] = len(venue_races)
            venue_positions = [r['place'] for r in venue_races if r['place'] > 0]
            if venue_positions:
                stats['venue_win_rate'] = sum(1 for p in venue_positions if p == 1) / len(venue_positions)
        
        # Time analysis
        times = [race['time'] for race in all_races if race['time'] > 0]
        if times:
            stats['best_time'] = min(times)
            stats['average_time'] = np.mean(times)
        
        # Weight consistency
        weights = [race['weight'] for race in all_races if race['weight'] > 0]
        if len(weights) >= 3:
            stats['weight_consistency'] = max(0, 1.0 - (np.std(weights) / np.mean(weights)))
        
        # Box preference analysis
        boxes = [race['box'] for race in all_races if race['box'] > 0]
        if boxes:
            # Calculate win rate by box position
            box_performance = {}
            for i, race in enumerate(all_races):
                if race['box'] > 0 and race['place'] > 0:
                    box = race['box']
                    if box not in box_performance:
                        box_performance[box] = []
                    box_performance[box].append(race['place'])
            
            # Find best performing box
            best_box_score = 0
            for box, positions in box_performance.items():
                avg_pos = np.mean(positions)
                box_score = max(0, 1.0 - (avg_pos - 1) / 7)  # Convert to 0-1 score
                best_box_score = max(best_box_score, box_score)
            
            stats['box_preference'] = best_box_score
        
        return stats
    
    def get_improved_ml_prediction_score(self, dog_name, dog_data, race_info):
        """Improved ML prediction without random noise"""
        try:
            historical_stats = self.calculate_comprehensive_historical_stats(dog_data, race_info)
            
            if historical_stats['total_races'] < self.min_races_for_reliable_prediction:
                return 0.4  # Low confidence for insufficient data, but not random
            
            # Calculate weighted score based on multiple factors
            factors = []
            weights = []
            
            # Win rate factor (40% weight)
            if historical_stats['win_rate'] > 0:
                factors.append(historical_stats['win_rate'])
                weights.append(0.4)
            
            # Place rate factor (25% weight)
            if historical_stats['place_rate'] > 0:
                place_score = historical_stats['place_rate']
                factors.append(place_score)
                weights.append(0.25)
            
            # Recent form trend (20% weight)
            form_score = 0.5 + (historical_stats['recent_form_trend'] * 0.2)
            form_score = max(0.1, min(0.9, form_score))
            factors.append(form_score)
            weights.append(0.2)
            
            # Consistency factor (15% weight)
            factors.append(historical_stats['consistency_score'])
            weights.append(0.15)
            
            if factors and weights:
                # Calculate weighted average
                weighted_score = np.average(factors, weights=weights)
                
                # Apply venue bonus/penalty
                if historical_stats['venue_experience'] >= 3:
                    venue_bonus = (historical_stats['venue_win_rate'] - 0.1) * 0.1
                    weighted_score += venue_bonus
                
                # Ensure score is within reasonable bounds
                return max(0.1, min(0.95, weighted_score))
            
            return 0.45  # Slightly below average if no meaningful factors
            
        except Exception as e:
            print(f"ML prediction error for {dog_name}: {e}")
            return 0.4
    
    def get_improved_traditional_analysis_score(self, dog_name, dog_data, race_info):
        """Improved traditional analysis without random noise"""
        try:
            historical_stats = self.calculate_comprehensive_historical_stats(dog_data, race_info)
            
            if historical_stats['total_races'] < self.min_races_for_reliable_prediction:
                return 0.35  # Lower than ML for insufficient data
            
            # Traditional handicapping factors
            score_components = []
            
            # Recent form (most important in traditional analysis)
            avg_position = historical_stats['average_position']
            position_score = max(0.1, 1.0 - (avg_position - 1) / 7)
            score_components.append(position_score * 0.4)
            
            # Win rate
            win_rate_score = historical_stats['win_rate'] * 2  # Scale up win rate
            score_components.append(min(0.3, win_rate_score))
            
            # Place rate 
            place_rate_score = historical_stats['place_rate'] * 0.8
            score_components.append(min(0.2, place_rate_score))
            
            # Consistency bonus
            consistency_bonus = historical_stats['consistency_score'] * 0.1
            score_components.append(consistency_bonus)
            
            # Recent trend adjustment
            if historical_stats['recent_form_trend'] > 0.1:  # Improving
                score_components.append(0.05)
            elif historical_stats['recent_form_trend'] < -0.1:  # Declining
                score_components.append(-0.05)
            
            # Venue experience bonus
            if historical_stats['venue_experience'] >= 3:
                venue_score = historical_stats['venue_win_rate'] * 0.1
                score_components.append(venue_score)
            
            final_score = sum(score_components)
            return max(0.1, min(0.9, final_score))
            
        except Exception as e:
            print(f"Traditional analysis error for {dog_name}: {e}")
            return 0.35
    
    def get_improved_weather_prediction_score(self, dog_name, dog_data, race_info):
        """Improved weather prediction without random noise"""
        try:
            # Start with traditional analysis as base
            base_score = self.get_improved_traditional_analysis_score(dog_name, dog_data, race_info)
            
            # Apply weather adjustments if available
            weather_adjustment = 1.0
            
            # Check for weather performance data
            if dog_data.get('weather_performance'):
                weather_data = dog_data['weather_performance']
                weather_adjustment = weather_data.get('adjustment_factor', 1.0)
            else:
                # Default weather considerations based on historical performance
                historical_stats = self.calculate_comprehensive_historical_stats(dog_data, race_info)
                
                # Dogs with better consistency might handle weather better
                if historical_stats['consistency_score'] > 0.7:
                    weather_adjustment = 1.05
                elif historical_stats['consistency_score'] < 0.3:
                    weather_adjustment = 0.95
            
            adjusted_score = base_score * weather_adjustment
            return max(0.1, min(0.9, adjusted_score))
            
        except Exception as e:
            print(f"Weather prediction error for {dog_name}: {e}")
            return 0.4
    
    def get_improved_enhanced_data_score(self, enhanced_data):
        """Improved enhanced data scoring"""
        try:
            if not enhanced_data:
                return 0.45
            
            score_components = []
            
            # PIR ratings analysis
            if enhanced_data.get('pir_ratings'):
                pir_values = [p['pir'] for p in enhanced_data['pir_ratings'] 
                             if p.get('pir') is not None and p['pir'] > 0]
                if pir_values:
                    avg_pir = np.mean(pir_values)
                    # PIR typically ranges from 1-100, with 50+ being good
                    pir_score = min(max((avg_pir - 30) / 70, 0.1), 0.9)
                    score_components.append(pir_score * 0.6)
            
            # Sectional times analysis
            if enhanced_data.get('sectional_times'):
                sectional_data = [s for s in enhanced_data['sectional_times'] 
                                if s.get('first_section') is not None and s['first_section'] > 0]
                if sectional_data:
                    sectionals = [s['first_section'] for s in sectional_data]
                    avg_sectional = np.mean(sectionals)
                    # Lower sectional times are better - typical range 5.0-6.5 seconds
                    if avg_sectional > 0:
                        sectional_score = max(0.1, min(0.9, (6.5 - avg_sectional) / 1.5))
                        score_components.append(sectional_score * 0.4)
            
            if score_components:
                return sum(score_components)
            else:
                return 0.45
                
        except Exception as e:
            print(f"Enhanced data scoring error: {e}")
            return 0.45
    
    def calculate_improved_weighted_final_score(self, prediction_scores, data_quality_score):
        """Calculate final score with improved weighting"""
        if not prediction_scores:
            return 0.4
        
        # Define weights based on data quality and method availability
        weights = {}
        total_weight = 0
        
        # ML system gets higher weight with good data quality
        if 'ml_system' in prediction_scores:
            ml_weight = 0.4 if data_quality_score > 0.6 else 0.3
            weights['ml_system'] = ml_weight
            total_weight += ml_weight
        
        # Traditional analysis is always available and reliable
        if 'traditional' in prediction_scores:
            trad_weight = 0.35
            weights['traditional'] = trad_weight
            total_weight += trad_weight
        
        # Weather enhanced gets moderate weight
        if 'weather_enhanced' in prediction_scores:
            weather_weight = 0.15
            weights['weather_enhanced'] = weather_weight
            total_weight += weather_weight
        
        # Enhanced data gets weight based on quality
        if 'enhanced_data' in prediction_scores:
            enhanced_weight = 0.1
            weights['enhanced_data'] = enhanced_weight
            total_weight += enhanced_weight
        
        # Calculate weighted average
        weighted_sum = 0
        for method, score in prediction_scores.items():
            if method in weights:
                weighted_sum += score * weights[method]
        
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = np.mean(list(prediction_scores.values()))
        
        # Apply data quality modifier
        quality_modifier = 0.8 + (data_quality_score * 0.4)  # Range: 0.8-1.2
        final_score *= quality_modifier
        
        return max(0.1, min(0.95, final_score))
    
    def determine_improved_confidence_level(self, prediction_scores, data_quality_score):
        """Determine confidence level based on data quality and score consistency"""
        if not prediction_scores:
            return 'VERY_LOW'
        
        # Check score consistency (lower variance = higher confidence)
        scores = list(prediction_scores.values())
        if len(scores) > 1:
            score_variance = np.var(scores)
            score_consistency = max(0, 1.0 - (score_variance * 10))  # Scale variance
        else:
            score_consistency = 0.5
        
        # Combine factors for confidence determination
        confidence_factors = [
            data_quality_score,
            score_consistency,
            min(1.0, len(prediction_scores) / 3)  # More methods = higher confidence
        ]
        
        overall_confidence = np.mean(confidence_factors)
        
        if overall_confidence >= 0.8:
            return 'HIGH'
        elif overall_confidence >= 0.6:
            return 'MEDIUM'
        elif overall_confidence >= 0.4:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _safe_int(self, value, default=0):
        """Safely convert value to integer"""
        try:
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                return default
            return int(float(str(value)))
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                return default
            return float(str(value))
        except (ValueError, TypeError):
            return default
