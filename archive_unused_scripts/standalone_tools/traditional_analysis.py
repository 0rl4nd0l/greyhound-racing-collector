#!/usr/bin/env python3
"""
Traditional Race Analysis Module
===============================

This module provides comprehensive traditional greyhound racing analysis metrics
that enrich machine learning training data. It implements proven handicapping
methods and statistical analysis techniques used by professional punters.

The traditional analysis focuses on:
- Historical performance metrics
- Form analysis and trends
- Class and grade analysis
- Track condition preferences
- Distance suitability
- Trainer and kennel statistics
- Recent activity patterns
- Consistency measurements

These traditional metrics are designed to work alongside ML predictions
to provide a comprehensive analysis framework.
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass


@dataclass
class TraditionalFactors:
    """Container for traditional analysis factors"""
    performance_score: float
    form_score: float
    class_score: float
    consistency_score: float
    fitness_score: float
    experience_score: float
    trainer_score: float
    track_condition_score: float
    distance_score: float
    recent_activity_score: float
    overall_traditional_score: float
    confidence_level: float
    key_factors: List[str]
    risk_factors: List[str]


class TraditionalRaceAnalyzer:
    """
    Comprehensive traditional greyhound racing analysis system.
    
    This analyzer implements time-tested handicapping methods and provides
    detailed traditional scoring that enriches ML training data.
    """
    
    def __init__(self, db_path: str = 'greyhound_data.db'):
        """
        Initialize the traditional race analyzer.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Traditional analysis weights - fine-tuned based on historical performance
        self.weights = {
            'performance': 0.25,      # Win rate, place rate, show rate
            'form': 0.20,            # Recent form trend and consistency
            'class': 0.15,           # Grade/class analysis
            'consistency': 0.12,     # Position and time consistency
            'fitness': 0.10,         # Recent activity and fitness indicators
            'experience': 0.08,      # Racing experience and maturity
            'trainer': 0.05,        # Trainer statistics and trends
            'track_condition': 0.03, # Track condition preferences
            'distance': 0.02        # Distance suitability
        }
        
        # Performance benchmarks for scoring
        self.benchmarks = {
            'excellent_win_rate': 0.30,
            'good_win_rate': 0.20,
            'average_win_rate': 0.15,
            'excellent_place_rate': 0.60,
            'good_place_rate': 0.45,
            'average_place_rate': 0.35,
            'excellent_avg_position': 3.0,
            'good_avg_position': 4.0,
            'average_avg_position': 5.0,
            'min_races_for_confidence': 5,
            'high_confidence_races': 15
        }
    
    def analyze_race(self, race_data: Dict, historical_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive traditional analysis for a race.
        
        Args:
            race_data: Dictionary containing race information and dog entries
            historical_data: Optional pre-loaded historical data
        
        Returns:
            Dictionary containing traditional analysis results for all dogs
        """
        try:
            results = {
                'race_info': race_data.get('race_info', {}),
                'analysis_timestamp': datetime.now().isoformat(),
                'traditional_predictions': [],
                'race_insights': {},
                'methodology': 'Traditional Handicapping Analysis v2.0'
            }
            
            dogs = race_data.get('dogs', [])
            if not dogs:
                self.logger.warning("No dogs found in race data")
                return results
            
            # Analyze each dog
            for dog_data in dogs:
                dog_name = dog_data.get('name', '').strip()
                if not dog_name:
                    continue
                
                # Get comprehensive dog analysis
                traditional_factors = self.analyze_dog(dog_name, race_data, historical_data)
                
                # Create traditional prediction entry
                prediction = {
                    'dog_name': dog_name,
                    'box_number': dog_data.get('box_number'),
                    'trainer': dog_data.get('trainer'),
                    'weight': dog_data.get('weight'),
                    'traditional_score': traditional_factors.overall_traditional_score,
                    'confidence_level': traditional_factors.confidence_level,
                    'breakdown': {
                        'performance_score': traditional_factors.performance_score,
                        'form_score': traditional_factors.form_score,
                        'class_score': traditional_factors.class_score,
                        'consistency_score': traditional_factors.consistency_score,
                        'fitness_score': traditional_factors.fitness_score,
                        'experience_score': traditional_factors.experience_score,
                        'trainer_score': traditional_factors.trainer_score,
                        'track_condition_score': traditional_factors.track_condition_score,
                        'distance_score': traditional_factors.distance_score,
                        'recent_activity_score': traditional_factors.recent_activity_score
                    },
                    'key_factors': traditional_factors.key_factors,
                    'risk_factors': traditional_factors.risk_factors,
                    'betting_recommendation': self._get_betting_recommendation(traditional_factors),
                    'ml_enrichment_features': self._extract_ml_features(traditional_factors, dog_data)
                }
                
                results['traditional_predictions'].append(prediction)
            
            # Sort predictions by traditional score
            results['traditional_predictions'].sort(
                key=lambda x: x['traditional_score'], reverse=True
            )
            
            # Add race insights
            results['race_insights'] = self._generate_race_insights(results['traditional_predictions'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in traditional race analysis: {e}")
            return {'error': str(e), 'traditional_predictions': []}
    
    def analyze_dog(self, dog_name: str, race_context: Dict, 
                   historical_data: Optional[Dict] = None) -> TraditionalFactors:
        """
        Perform comprehensive traditional analysis for a single dog.
        
        Args:
            dog_name: Clean dog name
            race_context: Race context information
            historical_data: Optional pre-loaded historical data
        
        Returns:
            TraditionalFactors object containing all analysis results
        """
        try:
            # Get or load historical data
            if historical_data and dog_name in historical_data:
                dog_stats = historical_data[dog_name]
            else:
                dog_stats = self._get_comprehensive_dog_stats(dog_name)
            
            # Calculate individual factor scores
            performance_score = self._calculate_performance_score(dog_stats)
            form_score = self._calculate_form_score(dog_stats)
            class_score = self._calculate_class_score(dog_stats, race_context)
            consistency_score = self._calculate_consistency_score(dog_stats)
            fitness_score = self._calculate_fitness_score(dog_stats)
            experience_score = self._calculate_experience_score(dog_stats)
            trainer_score = self._calculate_trainer_score(dog_stats)
            track_condition_score = self._calculate_track_condition_score(dog_stats, race_context)
            distance_score = self._calculate_distance_score(dog_stats, race_context)
            recent_activity_score = self._calculate_recent_activity_score(dog_stats)
            
            # Calculate weighted overall score
            overall_score = (
                performance_score * self.weights['performance'] +
                form_score * self.weights['form'] +
                class_score * self.weights['class'] +
                consistency_score * self.weights['consistency'] +
                fitness_score * self.weights['fitness'] +
                experience_score * self.weights['experience'] +
                trainer_score * self.weights['trainer'] +
                track_condition_score * self.weights['track_condition'] +
                distance_score * self.weights['distance']
            )
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(dog_stats)
            
            # Identify key factors and risks
            key_factors = self._identify_key_factors(dog_stats, race_context)
            risk_factors = self._identify_risk_factors(dog_stats, race_context)
            
            return TraditionalFactors(
                performance_score=performance_score,
                form_score=form_score,
                class_score=class_score,
                consistency_score=consistency_score,
                fitness_score=fitness_score,
                experience_score=experience_score,
                trainer_score=trainer_score,
                track_condition_score=track_condition_score,
                distance_score=distance_score,
                recent_activity_score=recent_activity_score,
                overall_traditional_score=min(overall_score, 1.0),
                confidence_level=confidence_level,
                key_factors=key_factors,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing dog {dog_name}: {e}")
            return self._get_default_factors()
    
    def _get_comprehensive_dog_stats(self, dog_name: str) -> Dict[str, Any]:
        """
        Get comprehensive historical statistics for a dog.
        
        Args:
            dog_name: Clean dog name
        
        Returns:
            Dictionary containing comprehensive dog statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all race results for the dog
            cursor.execute("""
                SELECT 
                    drd.finish_position, drd.individual_time, drd.margin, drd.starting_price,
                    rm.race_date, rm.venue, rm.distance, rm.grade, rm.track_condition,
                    drd.trainer_name, drd.weight, drd.box_number
                FROM dog_race_data drd
                JOIN race_metadata rm ON drd.race_id = rm.race_id
                WHERE UPPER(drd.dog_clean_name) = UPPER(?)
                ORDER BY rm.race_date DESC
            """, (dog_name,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return self._get_default_stats()
            
            # Process results into comprehensive statistics
            stats = self._process_race_results(results)
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting dog stats for {dog_name}: {e}")
            return self._get_default_stats()
    
    def _process_race_results(self, results: List[Tuple]) -> Dict[str, Any]:
        """
        Process raw race results into comprehensive statistics.
        
        Args:
            results: List of race result tuples
        
        Returns:
            Dictionary containing processed statistics
        """
        stats = {
            'races_count': len(results),
            'positions': [],
            'times': [],
            'margins': [],
            'odds': [],
            'recent_results': [],
            'trainers': [],
            'venues': [],
            'distances': [],
            'grades': [],
            'track_conditions': {},
            'win_count': 0,
            'place_count': 0,
            'show_count': 0
        }
        
        for i, result in enumerate(results):
            (position, time, margin, odds, race_date, venue, distance, 
             grade, track_condition, trainer, weight, box) = result
            
            # Store recent results (last 6 races)
            if i < 6:
                stats['recent_results'].append({
                    'position': position,
                    'time': time,
                    'race_date': race_date,
                    'venue': venue
                })
            
            # Count wins, places, shows
            if position:
                stats['positions'].append(position)
                if position == 1:
                    stats['win_count'] += 1
                if position <= 2:
                    stats['place_count'] += 1
                if position <= 3:
                    stats['show_count'] += 1
            
            # Store other data for analysis
            if time:
                stats['times'].append(float(time))
            if odds:
                stats['odds'].append(float(odds))
            if trainer:
                stats['trainers'].append(trainer)
            if venue:
                stats['venues'].append(venue)
            if distance:
                stats['distances'].append(distance)
            if grade:
                stats['grades'].append(grade)
            
            # Track condition analysis
            if track_condition:
                if track_condition not in stats['track_conditions']:
                    stats['track_conditions'][track_condition] = []
                if position:
                    stats['track_conditions'][track_condition].append(position)
        
        # Calculate derived statistics
        stats.update(self._calculate_derived_stats(stats))
        
        return stats
    
    def _calculate_derived_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived statistics from raw data."""
        derived = {}
        
        races_count = stats['races_count']
        positions = stats['positions']
        times = stats['times']
        
        if races_count > 0:
            # Basic rates
            derived['win_rate'] = stats['win_count'] / races_count
            derived['place_rate'] = stats['place_count'] / races_count
            derived['show_rate'] = stats['show_count'] / races_count
            
            # Position statistics
            if positions:
                derived['avg_position'] = np.mean(positions)
                derived['median_position'] = np.median(positions)
                derived['position_std'] = np.std(positions)
                derived['consistency'] = 1 / (1 + derived['position_std']) if derived['position_std'] > 0 else 1.0
                
                # Recent form (last 6 races)
                recent_positions = positions[:6] if len(positions) >= 6 else positions
                if len(recent_positions) >= 3:
                    derived['recent_avg_position'] = np.mean(recent_positions)
                    derived['form_trend'] = self._calculate_form_trend(recent_positions)
                else:
                    derived['recent_avg_position'] = derived['avg_position']
                    derived['form_trend'] = 0.0
            
            # Time statistics
            if times:
                derived['avg_time'] = np.mean(times)
                derived['best_time'] = min(times)
                derived['time_consistency'] = 1 / (1 + np.std(times)) if np.std(times) > 0 else 1.0
            
            # Trainer analysis
            if stats['trainers']:
                trainer_counts = {}
                for trainer in stats['trainers']:
                    trainer_counts[trainer] = trainer_counts.get(trainer, 0) + 1
                derived['primary_trainer'] = max(trainer_counts, key=trainer_counts.get)
                derived['trainer_stability'] = max(trainer_counts.values()) / races_count
            
            # Venue diversity
            derived['venue_count'] = len(set(stats['venues'])) if stats['venues'] else 0
            derived['venue_diversity'] = derived['venue_count'] / races_count if races_count > 0 else 0
            
            # Distance analysis
            if stats['distances']:
                distance_performance = {}
                for i, distance in enumerate(stats['distances']):
                    if distance and i < len(positions):
                        if distance not in distance_performance:
                            distance_performance[distance] = []
                        distance_performance[distance].append(positions[i])
                
                derived['distance_performance'] = distance_performance
                derived['preferred_distance'] = self._find_preferred_distance(distance_performance)
            
            # Track condition performance
            track_condition_stats = {}
            for condition, cond_positions in stats['track_conditions'].items():
                if cond_positions:
                    track_condition_stats[condition] = {
                        'avg_position': np.mean(cond_positions),
                        'races': len(cond_positions),
                        'win_rate': sum(1 for p in cond_positions if p == 1) / len(cond_positions)
                    }
            derived['track_condition_performance'] = track_condition_stats
            
            # Recent activity
            derived['recent_activity'] = self._calculate_recent_activity(stats['recent_results'])
        
        return derived
    
    def _calculate_performance_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate performance score based on win/place rates and average position."""
        if not dog_stats or dog_stats['races_count'] == 0:
            return 0.0
        
        win_rate = dog_stats.get('win_rate', 0)
        place_rate = dog_stats.get('place_rate', 0)
        avg_position = dog_stats.get('avg_position', 8)
        
        # Score components
        win_score = min(win_rate / self.benchmarks['excellent_win_rate'], 1.0)
        place_score = min(place_rate / self.benchmarks['excellent_place_rate'], 1.0)
        position_score = max(0, (8 - avg_position) / 8) if avg_position > 0 else 0
        
        # Weighted combination
        performance_score = (win_score * 0.4 + place_score * 0.35 + position_score * 0.25)
        
        return min(performance_score, 1.0)
    
    def _calculate_form_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate form score based on recent performance trend."""
        if not dog_stats or dog_stats['races_count'] < 3:
            return 0.5  # Neutral score for insufficient data
        
        form_trend = dog_stats.get('form_trend', 0)
        recent_avg_position = dog_stats.get('recent_avg_position', dog_stats.get('avg_position', 5))
        consistency = dog_stats.get('consistency', 0.5)
        
        # Form trend component (-2 to +2 scale, normalize to 0-1)
        trend_score = max(0, min(1, (form_trend + 2) / 4))
        
        # Recent position component
        recent_score = max(0, (8 - recent_avg_position) / 8) if recent_avg_position > 0 else 0
        
        # Consistency component
        consistency_bonus = consistency * 0.2
        
        form_score = (trend_score * 0.5 + recent_score * 0.3 + consistency_bonus)
        
        return min(form_score, 1.0)
    
    def _calculate_class_score(self, dog_stats: Dict[str, Any], race_context: Dict) -> float:
        """Calculate class score based on grade analysis."""
        if not dog_stats:
            return 0.5
        
        # Analyze grade progression and current class suitability
        grades = dog_stats.get('grades', [])
        current_grade = race_context.get('grade', 'Unknown')
        
        if not grades:
            return 0.5
        
        # Simple class scoring - can be enhanced with more sophisticated grade analysis
        grade_score = 0.6  # Base score
        
        # Bonus for grade consistency
        if current_grade in grades[:5]:  # Recently ran in this grade
            grade_score += 0.2
        
        # Performance in current grade
        grade_positions = []
        for i, grade in enumerate(grades):
            if grade == current_grade and i < len(dog_stats.get('positions', [])):
                grade_positions.append(dog_stats['positions'][i])
        
        if grade_positions:
            grade_avg_position = np.mean(grade_positions)
            if grade_avg_position <= 3:
                grade_score += 0.2
            elif grade_avg_position <= 5:
                grade_score += 0.1
        
        return min(grade_score, 1.0)
    
    def _calculate_consistency_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate consistency score."""
        if not dog_stats or dog_stats['races_count'] < 3:
            return 0.5
        
        consistency = dog_stats.get('consistency', 0.5)
        time_consistency = dog_stats.get('time_consistency', 0.5)
        
        # Combined consistency score
        consistency_score = (consistency * 0.6 + time_consistency * 0.4)
        
        return min(consistency_score, 1.0)
    
    def _calculate_fitness_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate fitness score based on recent activity and performance."""
        if not dog_stats:
            return 0.5
        
        recent_activity = dog_stats.get('recent_activity', {})
        days_since_last = recent_activity.get('days_since_last_race', 30)
        activity_score = recent_activity.get('activity_score', 0.5)
        
        # Optimal racing frequency bonus
        if 7 <= days_since_last <= 21:
            fitness_score = 0.8 + activity_score * 0.2
        elif days_since_last <= 7:
            fitness_score = 0.7 + activity_score * 0.2  # Might be tired
        elif days_since_last <= 35:
            fitness_score = 0.6 + activity_score * 0.3
        else:
            fitness_score = 0.4 + activity_score * 0.2  # Long layoff penalty
        
        return min(fitness_score, 1.0)
    
    def _calculate_experience_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate experience score."""
        if not dog_stats:
            return 0.3
        
        races_count = dog_stats['races_count']
        venue_diversity = dog_stats.get('venue_diversity', 0)
        
        # Experience curve - diminishing returns after 20 races
        experience_base = min(races_count / 20, 1.0)
        diversity_bonus = venue_diversity * 0.2
        
        experience_score = experience_base * 0.8 + diversity_bonus
        
        return min(experience_score, 1.0)
    
    def _calculate_trainer_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate trainer score based on stability and performance."""
        if not dog_stats:
            return 0.6
        
        trainer_stability = dog_stats.get('trainer_stability', 0.5)
        
        # Simple trainer scoring - can be enhanced with trainer statistics
        trainer_score = 0.5 + trainer_stability * 0.3
        
        return min(trainer_score, 1.0)
    
    def _calculate_track_condition_score(self, dog_stats: Dict[str, Any], race_context: Dict) -> float:
        """Calculate track condition suitability score."""
        if not dog_stats:
            return 0.6
        
        track_condition_performance = dog_stats.get('track_condition_performance', {})
        current_condition = race_context.get('track_condition', 'Good')
        
        if current_condition in track_condition_performance:
            condition_data = track_condition_performance[current_condition]
            if condition_data['races'] >= 2:
                avg_position = condition_data['avg_position']
                condition_score = max(0, (8 - avg_position) / 8)
                return min(condition_score + 0.2, 1.0)  # Bonus for having data
        
        return 0.6  # Default neutral score
    
    def _calculate_distance_score(self, dog_stats: Dict[str, Any], race_context: Dict) -> float:
        """Calculate distance suitability score."""
        if not dog_stats:
            return 0.6
        
        preferred_distance = dog_stats.get('preferred_distance')
        current_distance = race_context.get('distance')
        
        if preferred_distance and current_distance:
            if str(preferred_distance) == str(current_distance):
                return 0.9  # Strong bonus for preferred distance
            else:
                return 0.5  # Penalty for non-preferred distance
        
        return 0.6  # Default neutral score
    
    def _calculate_recent_activity_score(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate recent activity score."""
        if not dog_stats:
            return 0.5
        
        recent_activity = dog_stats.get('recent_activity', {})
        return recent_activity.get('activity_score', 0.5)
    
    def _calculate_confidence_level(self, dog_stats: Dict[str, Any]) -> float:
        """Calculate confidence level for the analysis."""
        if not dog_stats:
            return 0.2
        
        races_count = dog_stats['races_count']
        
        # Base confidence from sample size
        if races_count >= self.benchmarks['high_confidence_races']:
            base_confidence = 0.9
        elif races_count >= self.benchmarks['min_races_for_confidence']:
            base_confidence = 0.5 + (races_count - 5) * 0.04  # Gradual increase
        else:
            base_confidence = 0.2 + races_count * 0.06
        
        # Adjust for data quality
        recent_activity = dog_stats.get('recent_activity', {})
        days_since_last = recent_activity.get('days_since_last_race', 30)
        
        if days_since_last > 90:
            base_confidence *= 0.8  # Reduce confidence for long layoffs
        
        return min(base_confidence, 1.0)
    
    def _identify_key_factors(self, dog_stats: Dict[str, Any], race_context: Dict) -> List[str]:
        """Identify key positive factors for the dog."""
        factors = []
        
        if not dog_stats:
            return ["No historical data available"]
        
        # Performance factors
        win_rate = dog_stats.get('win_rate', 0)
        place_rate = dog_stats.get('place_rate', 0)
        avg_position = dog_stats.get('avg_position', 8)
        
        if win_rate >= self.benchmarks['excellent_win_rate']:
            factors.append(f"Excellent win rate ({win_rate:.1%})")
        elif win_rate >= self.benchmarks['good_win_rate']:
            factors.append(f"Good win rate ({win_rate:.1%})")
        
        if place_rate >= self.benchmarks['excellent_place_rate']:
            factors.append(f"Excellent place rate ({place_rate:.1%})")
        elif place_rate >= self.benchmarks['good_place_rate']:
            factors.append(f"Good place rate ({place_rate:.1%})")
        
        if avg_position <= self.benchmarks['excellent_avg_position']:
            factors.append(f"Consistently finishes well (avg: {avg_position:.1f})")
        
        # Form factors
        form_trend = dog_stats.get('form_trend', 0)
        if form_trend > 0.5:
            factors.append("Improving recent form")
        
        consistency = dog_stats.get('consistency', 0)
        if consistency > 0.8:
            factors.append("Very consistent performer")
        
        # Experience factors
        races_count = dog_stats['races_count']
        if races_count >= self.benchmarks['high_confidence_races']:
            factors.append(f"Highly experienced ({races_count} races)")
        
        # Activity factors
        recent_activity = dog_stats.get('recent_activity', {})
        days_since_last = recent_activity.get('days_since_last_race', 30)
        if days_since_last <= 14:
            factors.append("Recently active")
        
        # Track condition factors
        track_condition_performance = dog_stats.get('track_condition_performance', {})
        current_condition = race_context.get('track_condition', 'Good')
        if current_condition in track_condition_performance:
            condition_data = track_condition_performance[current_condition]
            if condition_data['races'] >= 2 and condition_data['avg_position'] <= 3.5:
                factors.append(f"Performs well on {current_condition.lower()} tracks")
        
        return factors
    
    def _identify_risk_factors(self, dog_stats: Dict[str, Any], race_context: Dict) -> List[str]:
        """Identify risk factors for the dog."""
        risks = []
        
        if not dog_stats:
            return ["No historical data for assessment"]
        
        # Performance risks
        win_rate = dog_stats.get('win_rate', 0)
        place_rate = dog_stats.get('place_rate', 0)
        avg_position = dog_stats.get('avg_position', 8)
        races_count = dog_stats['races_count']
        
        if win_rate < 0.10 and races_count > 5:
            risks.append("Low win rate")
        
        if place_rate < 0.30 and races_count > 5:
            risks.append("Struggles to place")
        
        if avg_position > 6:
            risks.append("Often finishes poorly")
        
        # Experience risks
        if races_count < 3:
            risks.append("Limited racing experience")
        
        # Form risks
        form_trend = dog_stats.get('form_trend', 0)
        if form_trend < -0.5:
            risks.append("Declining recent form")
        
        consistency = dog_stats.get('consistency', 0)
        if consistency < 0.5:
            risks.append("Inconsistent performance")
        
        # Activity risks
        recent_activity = dog_stats.get('recent_activity', {})
        days_since_last = recent_activity.get('days_since_last_race', 30)
        if days_since_last > 60:
            risks.append("Long layoff from racing")
        elif days_since_last < 5:
            risks.append("Very recent race (possible fatigue)")
        
        return risks
    
    def _get_betting_recommendation(self, factors: TraditionalFactors) -> str:
        """Get betting recommendation based on traditional factors."""
        score = factors.overall_traditional_score
        confidence = factors.confidence_level
        
        if score >= 0.7 and confidence >= 0.8:
            return "Strong Win"
        elif score >= 0.6 and confidence >= 0.7:
            return "Win/Place"
        elif score >= 0.4 and confidence >= 0.6:
            return "Place Only"
        elif score >= 0.3:
            return "Each-Way"
        else:
            return "Avoid"
    
    def _extract_ml_features(self, factors: TraditionalFactors, dog_data: Dict) -> Dict[str, float]:
        """Extract features for ML training enrichment."""
        return {
            'traditional_overall_score': factors.overall_traditional_score,
            'traditional_performance_score': factors.performance_score,
            'traditional_form_score': factors.form_score,
            'traditional_class_score': factors.class_score,
            'traditional_consistency_score': factors.consistency_score,
            'traditional_fitness_score': factors.fitness_score,
            'traditional_experience_score': factors.experience_score,
            'traditional_trainer_score': factors.trainer_score,
            'traditional_track_condition_score': factors.track_condition_score,
            'traditional_distance_score': factors.distance_score,
            'traditional_confidence_level': factors.confidence_level,
            'traditional_key_factors_count': len(factors.key_factors),
            'traditional_risk_factors_count': len(factors.risk_factors)
        }
    
    def _generate_race_insights(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Generate insights about the race based on traditional analysis."""
        if not predictions:
            return {}
        
        total_dogs = len(predictions)
        high_confidence_dogs = len([p for p in predictions if p['confidence_level'] >= 0.7])
        strong_contenders = len([p for p in predictions if p['traditional_score'] >= 0.6])
        
        top_dog = predictions[0]
        score_gap = top_dog['traditional_score'] - predictions[1]['traditional_score'] if len(predictions) > 1 else 0
        
        return {
            'total_analyzed': total_dogs,
            'high_confidence_selections': high_confidence_dogs,
            'strong_contenders': strong_contenders,
            'favorite': {
                'name': top_dog['dog_name'],
                'score': top_dog['traditional_score'],
                'confidence': top_dog['confidence_level']
            },
            'competitiveness': 'Low' if score_gap > 0.2 else 'High',
            'data_quality': 'Good' if high_confidence_dogs >= total_dogs * 0.6 else 'Limited'
        }
    
    def _calculate_form_trend(self, recent_positions: List[int]) -> float:
        """Calculate form trend from recent positions."""
        if len(recent_positions) < 3:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(recent_positions)))
        y = [8 - pos for pos in recent_positions]  # Invert so higher is better
        
        # Calculate slope
        n = len(x)
        slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi ** 2 for xi in x) - sum(x) ** 2)
        
        # Normalize to -2 to +2 range
        return max(-2, min(2, slope))
    
    def _find_preferred_distance(self, distance_performance: Dict[str, List[int]]) -> Optional[str]:
        """Find the dog's preferred distance based on performance."""
        if not distance_performance:
            return None
        
        best_distance = None
        best_avg_position = 8
        
        for distance, positions in distance_performance.items():
            if len(positions) >= 2:  # Minimum sample size
                avg_position = np.mean(positions)
                if avg_position < best_avg_position:
                    best_avg_position = avg_position
                    best_distance = distance
        
        return best_distance
    
    def _calculate_recent_activity(self, recent_results: List[Dict]) -> Dict[str, Any]:
        """Calculate recent activity metrics."""
        if not recent_results:
            return {'days_since_last_race': 365, 'activity_score': 0.0}
        
        # Get days since last race
        try:
            last_race_date = datetime.strptime(recent_results[0]['race_date'], '%Y-%m-%d')
            days_since_last = (datetime.now() - last_race_date).days
        except:
            days_since_last = 30
        
        # Calculate activity score (optimal is 7-21 days)
        if 7 <= days_since_last <= 21:
            activity_score = 1.0
        elif days_since_last <= 7:
            activity_score = 0.8
        elif days_since_last <= 35:
            activity_score = 0.7
        elif days_since_last <= 60:
            activity_score = 0.5
        else:
            activity_score = 0.3
        
        return {
            'days_since_last_race': days_since_last,
            'activity_score': activity_score
        }
    
    def _get_default_factors(self) -> TraditionalFactors:
        """Get default factors for dogs with no data."""
        return TraditionalFactors(
            performance_score=0.3,
            form_score=0.3,
            class_score=0.5,
            consistency_score=0.3,
            fitness_score=0.5,
            experience_score=0.2,
            trainer_score=0.5,
            track_condition_score=0.5,
            distance_score=0.5,
            recent_activity_score=0.5,
            overall_traditional_score=0.35,
            confidence_level=0.2,
            key_factors=["No historical data available"],
            risk_factors=["Unknown performance history"]
        )
    
    def _get_default_stats(self) -> Dict[str, Any]:
        """Get default stats for dogs with no data."""
        return {
            'races_count': 0,
            'win_rate': 0.0,
            'place_rate': 0.0,
            'show_rate': 0.0,
            'avg_position': 5.0,
            'consistency': 0.5,
            'form_trend': 0.0,
            'recent_activity': {'days_since_last_race': 30, 'activity_score': 0.5},
            'trainer_stability': 0.5,
            'venue_diversity': 0.0
        }


# Helper functions for integration with existing systems
def calculate_traditional_score(dog_name: str, race_context: Dict, 
                              db_path: str = 'greyhound_data.db') -> float:
    """
    Convenience function to get traditional score for a single dog.
    
    Args:
        dog_name: Clean dog name
        race_context: Race context information
        db_path: Path to database
    
    Returns:
        Traditional score (0.0 to 1.0)
    """
    analyzer = TraditionalRaceAnalyzer(db_path)
    factors = analyzer.analyze_dog(dog_name, race_context)
    return factors.overall_traditional_score


def get_traditional_ml_features(dog_name: str, race_context: Dict,
                              db_path: str = 'greyhound_data.db') -> Dict[str, float]:
    """
    Get traditional analysis features for ML training.
    
    Args:
        dog_name: Clean dog name
        race_context: Race context information
        db_path: Path to database
    
    Returns:
        Dictionary of features for ML training
    """
    analyzer = TraditionalRaceAnalyzer(db_path)
    factors = analyzer.analyze_dog(dog_name, race_context)
    return analyzer._extract_ml_features(factors, {'name': dog_name})


if __name__ == "__main__":
    # Example usage
    analyzer = TraditionalRaceAnalyzer()
    
    # Test with sample data
    sample_race = {
        'race_info': {'venue': 'DAPT', 'distance': '500m', 'grade': 'Grade 5'},
        'track_condition': 'Good',
        'dogs': [
            {'name': 'SAMPLE DOG', 'box_number': 1, 'trainer': 'J. Smith', 'weight': '30.5kg'}
        ]
    }
    
    results = analyzer.analyze_race(sample_race)
    print("Traditional Analysis Results:")
    print(f"Total predictions: {len(results.get('traditional_predictions', []))}")
    
    if results.get('traditional_predictions'):
        top_pick = results['traditional_predictions'][0]
        print(f"Top pick: {top_pick['dog_name']} - Score: {top_pick['traditional_score']:.3f}")
