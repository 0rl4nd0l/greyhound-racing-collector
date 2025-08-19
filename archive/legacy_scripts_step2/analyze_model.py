from model_registry import get_model_registry
import numpy as np
import pandas as pd

# Get the global model registry
registry = get_model_registry()

# Get the best model
best_model_result = registry.get_best_model()

if best_model_result:
    model, scaler, metadata = best_model_result
    print(f"Model Analysis for {metadata.model_id}")
    print(f"Type: {metadata.model_name}")
    print(f"Feature Names: {metadata.feature_names}")
    
    # Test different box position combinations to understand decision boundaries
    box_positions = [(1.5, 0.42), (4.5, 0.22), (6.8, 0.08)]  # (avg, win_rate)
    form_levels = [(1.8, 0.85), (3.5, 0.6), (5.2, 0.3)]  # (form, consistency)
    
    print("\nDecision Surface Analysis:")
    print("---------------------------")
    
    base_features = {
        'weighted_recent_form': 3.5,      # Average form
        'speed_trend': 0.0,              # Neutral trend
        'speed_consistency': 0.6,        # Average consistency
        'venue_win_rate': 0.15,          # Average venue performance
        'venue_avg_position': 4.0,       # Mid-pack
        'venue_experience': 6,           # Some experience
        'distance_win_rate': 0.12,       # Average at distance
        'distance_avg_time': 30.2,       # Average time
        'box_position_win_rate': 0.15,   # Average from box
        'box_position_avg': 4.5,         # Mid-pack box
        'recent_momentum': 0.5,          # Neutral momentum
        'competitive_level': 0.5,        # Mid-tier
        'position_consistency': 0.6,     # Average consistency
        'top_3_rate': 0.35,             # Sometimes places
        'break_quality': 0.6             # Average break
    }
    
    # Test box position influence
    print("\nBox Position Influence:")
    for avg, win_rate in box_positions:
        features = base_features.copy()
        features['box_position_avg'] = avg
        features['box_position_win_rate'] = win_rate
        
        X = pd.DataFrame([features], columns=metadata.feature_names)
        pred = model.predict_proba(X)[0][1]
        print(f"Box avg={avg}, win_rate={win_rate}: {pred:.4f}")
    
    # Test form influence
    print("\nForm Influence:")
    for form, consistency in form_levels:
        features = base_features.copy()
        features['weighted_recent_form'] = form
        features['speed_consistency'] = consistency
        
        X = pd.DataFrame([features], columns=metadata.feature_names)
        pred = model.predict_proba(X)[0][1]
        print(f"Form={form}, consistency={consistency}: {pred:.4f}")
    
    # Check decision path for strong contender
    if hasattr(model, 'decision_path'):
        strong_features = {
            'box_position_avg': 1.5,
            'box_position_win_rate': 0.42,
            'weighted_recent_form': 1.8,
            'speed_consistency': 0.85,
            'venue_win_rate': 0.40,
            'venue_avg_position': 2.5,
            'venue_experience': 12,
            'distance_win_rate': 0.35,
            'distance_avg_time': 29.8,
            'box_position_win_rate': 0.42,
            'recent_momentum': 0.85,
            'competitive_level': 0.8,
            'position_consistency': 0.85,
            'top_3_rate': 0.65,
            'break_quality': 0.9
        }
        X = pd.DataFrame([strong_features], columns=metadata.feature_names)
        path = model.decision_path(X)
        print("\nStrong Contender Decision Path:")
        print(f"Path length: {len(path.indices)}")
else:
    print("No model found in registry")
