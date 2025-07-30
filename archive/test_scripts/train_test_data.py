import numpy as np
import pandas as pd
from advanced_ml_system_v2 import AdvancedMLSystemV2

# Set random seed for reproducibility
np.random.seed(42)

# Create properly normalized training data with realistic patterns
features_list = []

def normalize_to_01(value, min_val, max_val):
    """Normalize value to 0-1 range"""
    return (value - min_val) / (max_val - min_val)

for i in range(300):  # More training samples
    # Create patterns that correlate with winning
    is_winner = np.random.random() < 0.125  # 12.5% win rate (1 in 8)
    
    if is_winner:
        # Good performers - all values normalized to 0-1 range
        raw_form = np.random.uniform(1.0, 3.0)  # Better form (1-6 scale)
        features = {
            'weighted_recent_form': normalize_to_01(raw_form, 1.0, 6.0),  # Normalize form to 0-1
            'speed_trend': normalize_to_01(np.random.uniform(-0.15, 0.05), -0.2, 0.2),  # Improving trend
            'speed_consistency': np.random.uniform(0.6, 0.9),  # Already 0-1
            'venue_win_rate': np.random.uniform(0.15, 0.45),  # Already 0-1
            'venue_avg_position': normalize_to_01(np.random.uniform(2.0, 4.0), 1.0, 8.0),  # Better positions
            'venue_experience': normalize_to_01(np.random.randint(5, 20), 0, 25),  # More experienced
            'distance_win_rate': np.random.uniform(0.1, 0.4),  # Already 0-1
            'distance_avg_time': normalize_to_01(np.random.uniform(29.8, 30.4), 29.0, 32.0),  # Faster times
            'box_position_win_rate': np.random.uniform(0.15, 0.5),  # Already 0-1
            'box_position_avg': normalize_to_01(np.random.uniform(1.0, 5.0), 1.0, 8.0),  # Better box positions
            'recent_momentum': np.random.uniform(0.6, 0.95),  # Already 0-1
            'competitive_level': np.random.uniform(0.5, 0.9),  # Already 0-1
            'position_consistency': np.random.uniform(0.6, 0.9),  # Already 0-1
            'top_3_rate': np.random.uniform(0.4, 0.8),  # Already 0-1
            'break_quality': np.random.uniform(0.6, 0.95)  # Already 0-1
        }
    else:
        # Poor/average performers - all values normalized to 0-1 range
        raw_form = np.random.uniform(2.5, 6.0)  # Worse form
        features = {
            'weighted_recent_form': normalize_to_01(raw_form, 1.0, 6.0),  # Normalize form to 0-1
            'speed_trend': normalize_to_01(np.random.uniform(-0.1, 0.2), -0.2, 0.2),  # Variable trend
            'speed_consistency': np.random.uniform(0.3, 0.7),  # Already 0-1
            'venue_win_rate': np.random.uniform(0.02, 0.25),  # Already 0-1
            'venue_avg_position': normalize_to_01(np.random.uniform(3.5, 7.0), 1.0, 8.0),  # Worse positions
            'venue_experience': normalize_to_01(np.random.randint(1, 15), 0, 25),  # Variable experience
            'distance_win_rate': np.random.uniform(0.02, 0.3),  # Already 0-1
            'distance_avg_time': normalize_to_01(np.random.uniform(30.2, 31.2), 29.0, 32.0),  # Slower times
            'box_position_win_rate': np.random.uniform(0.05, 0.35),  # Already 0-1
            'box_position_avg': normalize_to_01(np.random.uniform(2.0, 8.0), 1.0, 8.0),  # Variable box positions
            'recent_momentum': np.random.uniform(0.2, 0.7),  # Already 0-1
            'competitive_level': np.random.uniform(0.2, 0.7),  # Already 0-1
            'position_consistency': np.random.uniform(0.3, 0.7),  # Already 0-1
            'top_3_rate': np.random.uniform(0.1, 0.5),  # Already 0-1
            'break_quality': np.random.uniform(0.3, 0.8)  # Already 0-1
        }
    
    features_list.append({
        'target': 1 if is_winner else 0,
        'features': features
    })

# Initialize ML system without auto-loading
ml_system = AdvancedMLSystemV2(skip_auto_load=True)

# Prepare data
X, y, feature_columns, df = ml_system.prepare_training_data(features_list)

# Train models
if X is not None and y is not None:
    ml_system.train_advanced_models(X, y, feature_columns)
    
    # Save the trained models
    ml_system.save_models("advanced_ml_model_test")
