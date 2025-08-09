from advanced_ml_system_v2 import AdvancedMLSystemV2
import numpy as np

# Initialize ML system without auto-loading to use fresh ensemble
ml_system = AdvancedMLSystemV2(skip_auto_load=True)

# Load the most recent trained model
import glob
import os

# Find the most recent test model file
model_files = glob.glob("advanced_ml_model_test_*.joblib")
if model_files:
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 Loading latest trained model: {latest_model}")
    ml_system.load_models(latest_model)
else:
    print("❌ No trained test models found!")
    exit(1)

print("🔍 Testing newly trained model with scaling fixes")
print("=" * 60)

def normalize_to_01(value, min_val, max_val):
    """Normalize value to 0-1 range"""
    return (value - min_val) / (max_val - min_val)

# Test cases with varying quality levels - all values normalized to 0-1
test_cases = [
    {
        'name': 'Strong Contender',
        'features': {
            'weighted_recent_form': normalize_to_01(1.5, 1.0, 6.0),  # Better form
            'speed_trend': normalize_to_01(-0.1, -0.2, 0.2),  # Improving
            'speed_consistency': 0.8,  # Already 0-1
            'venue_win_rate': 0.35,  # Already 0-1
            'venue_avg_position': normalize_to_01(2.5, 1.0, 8.0),  # Good position
            'venue_experience': normalize_to_01(12, 0, 25),  # Experienced
            'distance_win_rate': 0.3,  # Already 0-1
            'distance_avg_time': normalize_to_01(29.9, 29.0, 32.0),  # Fast time
            'box_position_win_rate': 0.4,  # Already 0-1
            'box_position_avg': normalize_to_01(2.0, 1.0, 8.0),  # Good box
            'recent_momentum': 0.8,  # Already 0-1
            'competitive_level': 0.7,  # Already 0-1
            'position_consistency': 0.8,  # Already 0-1
            'top_3_rate': 0.6,  # Already 0-1
            'break_quality': 0.85,  # Already 0-1
            'data_quality': 0.9  # Already 0-1
        }
    },
    {
        'name': 'Average Performer',
        'features': {
            'weighted_recent_form': normalize_to_01(3.5, 1.0, 6.0),  # Average form
            'speed_trend': normalize_to_01(0.0, -0.2, 0.2),  # Neutral
            'speed_consistency': 0.6,  # Already 0-1
            'venue_win_rate': 0.2,  # Already 0-1
            'venue_avg_position': normalize_to_01(4.0, 1.0, 8.0),  # Mid position
            'venue_experience': normalize_to_01(7, 0, 25),  # Some experience
            'distance_win_rate': 0.18,  # Already 0-1
            'distance_avg_time': normalize_to_01(30.3, 29.0, 32.0),  # Average time
            'box_position_win_rate': 0.22,  # Already 0-1
            'box_position_avg': normalize_to_01(4.5, 1.0, 8.0),  # Mid box
            'recent_momentum': 0.5,  # Already 0-1
            'competitive_level': 0.5,  # Already 0-1
            'position_consistency': 0.6,  # Already 0-1
            'top_3_rate': 0.35,  # Already 0-1
            'break_quality': 0.6,  # Already 0-1
            'data_quality': 0.7  # Already 0-1
        }
    },
    {
        'name': 'Weak Competitor',
        'features': {
            'weighted_recent_form': normalize_to_01(5.5, 1.0, 6.0),  # Poor form
            'speed_trend': normalize_to_01(0.15, -0.2, 0.2),  # Declining
            'speed_consistency': 0.4,  # Already 0-1
            'venue_win_rate': 0.08,  # Already 0-1
            'venue_avg_position': normalize_to_01(6.0, 1.0, 8.0),  # Poor position
            'venue_experience': normalize_to_01(3, 0, 25),  # Inexperienced
            'distance_win_rate': 0.06,  # Already 0-1
            'distance_avg_time': normalize_to_01(30.8, 29.0, 32.0),  # Slow time
            'box_position_win_rate': 0.1,  # Already 0-1
            'box_position_avg': normalize_to_01(7.0, 1.0, 8.0),  # Poor box
            'recent_momentum': 0.3,  # Already 0-1
            'competitive_level': 0.35,  # Already 0-1
            'position_consistency': 0.4,  # Already 0-1
            'top_3_rate': 0.2,  # Already 0-1
            'break_quality': 0.4,  # Already 0-1
            'data_quality': 0.5  # Already 0-1
        }
    }
]

print(f"Testing {len(test_cases)} different performance profiles:")
print()

for i, test_case in enumerate(test_cases, 1):
    print(f"{i}. {test_case['name']}:")
    print("-" * 40)
    
    prediction = ml_system.predict_with_ensemble(test_case['features'])
    confidence = ml_system.generate_prediction_confidence(test_case['features'])
    
    print(f"   Final Prediction: {prediction:.4f}")
    print(f"   Confidence: {confidence:.4f}")
    print()

# Test gradient across different box positions
print("\n🎯 Box Position Impact Analysis:")
print("=" * 40)

base_features = {
    'weighted_recent_form': normalize_to_01(3.0, 1.0, 6.0),  # Mid form
    'speed_trend': normalize_to_01(0.0, -0.2, 0.2),  # Neutral
    'speed_consistency': 0.7,  # Already 0-1
    'venue_win_rate': 0.25,  # Already 0-1
    'venue_avg_position': normalize_to_01(3.5, 1.0, 8.0),  # Good-mid position
    'venue_experience': normalize_to_01(8, 0, 25),  # Some experience
    'distance_win_rate': 0.2,  # Already 0-1
    'distance_avg_time': normalize_to_01(30.2, 29.0, 32.0),  # Average time
    'recent_momentum': 0.6,  # Already 0-1
    'competitive_level': 0.6,  # Already 0-1
    'position_consistency': 0.7,  # Already 0-1
    'top_3_rate': 0.4,  # Already 0-1
    'break_quality': 0.7,  # Already 0-1
    'data_quality': 0.8  # Already 0-1
}

for box_pos in [1.5, 3.0, 4.5, 6.0, 7.5]:
    features = base_features.copy()
    features['box_position_avg'] = normalize_to_01(box_pos, 1.0, 8.0)  # Normalize box position
    features['box_position_win_rate'] = max(0.05, 0.45 - (box_pos * 0.05))  # Already 0-1
    
    prediction = ml_system.predict_with_ensemble(features)
    print(f"Box {box_pos}: {prediction:.4f} (win_rate: {features['box_position_win_rate']:.3f})")
