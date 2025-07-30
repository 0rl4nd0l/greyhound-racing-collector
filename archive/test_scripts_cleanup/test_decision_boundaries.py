from model_registry import get_model_registry
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Get the global model registry
registry = get_model_registry()
best_model_result = registry.get_best_model()

if best_model_result:
    model, scaler, metadata = best_model_result
    print(f"Model Analysis for {metadata.model_id}")
    print(f"Type: {metadata.model_name}")
    
    # Base feature set (average performer)
    base_features = {
        'weighted_recent_form': 3.5,
        'speed_trend': 0.0,
        'speed_consistency': 0.6,
        'venue_win_rate': 0.15,
        'venue_avg_position': 4.0,
        'venue_experience': 6,
        'distance_win_rate': 0.12,
        'distance_avg_time': 30.2,
        'box_position_win_rate': 0.15,
        'box_position_avg': 4.5,
        'recent_momentum': 0.5,
        'competitive_level': 0.5,
        'position_consistency': 0.6,
        'top_3_rate': 0.35,
        'break_quality': 0.6
    }
    
    print("\nBox Position Response Curve:")
    print("-" * 40)
    
    # Test a range of box positions
    box_positions = np.linspace(1.0, 8.0, 15)
    for box_avg in box_positions:
        features = base_features.copy()
        features['box_position_avg'] = box_avg
        features['box_position_win_rate'] = max(0.05, 0.45 - (box_avg * 0.05))  # Decreasing win rate with box
        
        X = pd.DataFrame([features], columns=metadata.feature_names)
        pred = model.predict_proba(X)[0][1]
        print(f"Box {box_avg:.1f} (win_rate={features['box_position_win_rate']:.3f}): {pred:.4f}")
    
    print("\nForm Impact Testing:")
    print("-" * 40)
    
    # Reset box position to good but not exceptional
    base_features['box_position_avg'] = 2.5
    base_features['box_position_win_rate'] = 0.32
    
    # Test range of form values
    form_values = np.linspace(1.0, 6.0, 11)
    for form in form_values:
        features = base_features.copy()
        features['weighted_recent_form'] = form
        features['speed_consistency'] = max(0.3, 1.0 - (form * 0.1))  # Better form = better consistency
        
        X = pd.DataFrame([features], columns=metadata.feature_names)
        pred = model.predict_proba(X)[0][1]
        print(f"Form {form:.1f} (consistency={features['speed_consistency']:.3f}): {pred:.4f}")
    
    print("\nFull Feature Set Gradients:")
    print("-" * 40)
    
    # Create a progression from very good to very poor
    quality_levels = np.linspace(0.0, 1.0, 11)  # 0 = best, 1 = worst
    
    for quality in quality_levels:
        features = {
            'box_position_avg': 1.0 + (quality * 7.0),  # 1.0 to 8.0
            'box_position_win_rate': 0.45 - (quality * 0.4),  # 0.45 to 0.05
            'weighted_recent_form': 1.5 + (quality * 4.5),  # 1.5 to 6.0
            'speed_trend': -0.2 + (quality * 0.4),  # -0.2 to 0.2
            'speed_consistency': 0.9 - (quality * 0.6),  # 0.9 to 0.3
            'venue_win_rate': 0.4 - (quality * 0.35),  # 0.4 to 0.05
            'venue_avg_position': 2.0 + (quality * 4.5),  # 2.0 to 6.5
            'venue_experience': 15 - (quality * 13),  # 15 to 2
            'distance_win_rate': 0.35 - (quality * 0.3),  # 0.35 to 0.05
            'distance_avg_time': 29.8 + (quality * 1.2),  # 29.8 to 31.0
            'recent_momentum': 0.9 - (quality * 0.7),  # 0.9 to 0.2
            'competitive_level': 0.85 - (quality * 0.55),  # 0.85 to 0.3
            'position_consistency': 0.9 - (quality * 0.6),  # 0.9 to 0.3
            'top_3_rate': 0.7 - (quality * 0.55),  # 0.7 to 0.15
            'break_quality': 0.95 - (quality * 0.65)  # 0.95 to 0.3
        }
        
        X = pd.DataFrame([features], columns=metadata.feature_names)
        
        # Try both raw and scaled prediction
        raw_pred = model.predict_proba(X)[0][1]
        
        X_scaled = pd.DataFrame(scaler.transform(X), columns=metadata.feature_names)
        scaled_pred = model.predict_proba(X_scaled)[0][1]
        
        print(f"Quality {1.0-quality:.1f}: raw={raw_pred:.4f}, scaled={scaled_pred:.4f}")
else:
    print("No model found in registry")
