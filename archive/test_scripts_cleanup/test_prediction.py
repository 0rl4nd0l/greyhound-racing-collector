import numpy as np
import pandas as pd
from advanced_ml_system_v2 import AdvancedMLSystemV2

def print_prediction_details(ml_system, features, case_name):
    prediction = ml_system.predict_with_ensemble(features)
    confidence = ml_system.generate_prediction_confidence(features)
    print(f"\n=== {case_name} ===")
    print(f"Prediction Score: {prediction:.4f}")
    print(f"Confidence Score: {confidence:.4f}")
    return prediction, confidence

# Initialize the ML system
print("Initializing ML System...\n")
ml_system = AdvancedMLSystemV2()

# Test Case 1: Strong Contender (Box 1)
strong_contender = {
    'box_position_avg': 1.2,  # Exceptional box average
    'box_position_win_rate': 0.45,  # High win rate from box 1
    'weighted_recent_form': 2.0,  # Very good form
    'speed_consistency': 0.88,  # Very consistent
    'distance_avg_time': 29.8,  # Good time
    'speed_trend': -0.15,  # Improving times
    'venue_win_rate': 0.42,  # Strong at venue
    'venue_avg_position': 2.2,  # Strong average position
    'venue_experience': 15,  # Extensive experience
    'distance_win_rate': 0.38,  # Strong at distance
    'recent_momentum': 0.92,  # Excellent momentum
    'competitive_level': 0.85,  # High class
    'position_consistency': 0.90,  # Very consistent
    'top_3_rate': 0.72,  # Often places
    'break_quality': 0.95  # Fast breaker
}

# Test Case 2: Good Performer (Box 3)
average_performer = {
    'box_position_avg': 2.8,  # Good box average
    'box_position_win_rate': 0.28,  # Good win rate from box 3
    'weighted_recent_form': 2.8,  # Good form
    'speed_consistency': 0.75,  # Consistent
    'distance_avg_time': 30.0,  # Good time
    'speed_trend': -0.05,  # Slightly improving
    'venue_win_rate': 0.25,  # Good at venue
    'venue_avg_position': 3.2,  # Good average position
    'venue_experience': 10,  # Good experience
    'distance_win_rate': 0.22,  # Good at distance
    'recent_momentum': 0.70,  # Good momentum
    'competitive_level': 0.65,  # Above average class
    'position_consistency': 0.72,  # Consistent
    'top_3_rate': 0.48,  # Often places
    'break_quality': 0.75  # Good break
}

# Test Case 3: Outsider (Box 6)
longshot = {
    'box_position_avg': 4.2,  # Below average box position
    'box_position_win_rate': 0.18,  # Lower win rate from box 6
    'weighted_recent_form': 3.8,  # Average form
    'speed_consistency': 0.55,  # Moderate consistency
    'distance_avg_time': 30.4,  # Average time
    'speed_trend': 0.05,  # Stable times
    'venue_win_rate': 0.15,  # Average at venue
    'venue_avg_position': 4.5,  # Mid-pack average
    'venue_experience': 8,  # Some experience
    'distance_win_rate': 0.12,  # Average at distance
    'recent_momentum': 0.45,  # Moderate momentum
    'competitive_level': 0.45,  # Average class
    'position_consistency': 0.50,  # Moderate consistency
    'top_3_rate': 0.35,  # Sometimes places
    'break_quality': 0.55  # Average break
}

# Run predictions
print("Running Test Predictions...")

score1, conf1 = print_prediction_details(ml_system, strong_contender, "Strong Contender")
score2, conf2 = print_prediction_details(ml_system, average_performer, "Average Performer")
score3, conf3 = print_prediction_details(ml_system, longshot, "Longshot")

# Verify differentiation
score_range = max(score1, score2, score3) - min(score1, score2, score3)
print(f"\nScore Differentiation Analysis:")
print(f"Score Range: {score_range:.4f}")
print(f"Expected order maintained: {score1 > score2 > score3}")
