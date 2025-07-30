from model_registry import get_model_registry
import numpy as np
import pandas as pd

# Get the global model registry
registry = get_model_registry()
best_model_result = registry.get_best_model()

if best_model_result:
    model, scaler, metadata = best_model_result
    print(f"Model Analysis for {metadata.model_id}")
    print(f"Feature names: {metadata.feature_names}")
    
    # Check scaler properties
    if hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
        print("\nScaler Statistics:")
        print("-" * 40)
        for i, feature in enumerate(metadata.feature_names):
            center = scaler.center_[i] if hasattr(scaler, 'center_') else 'N/A'
            scale = scaler.scale_[i] if hasattr(scaler, 'scale_') else 'N/A'
            print(f"{feature:25s}: center={center:8.4f}, scale={scale:8.4f}")
    
    # Test with a reasonable feature set
    test_features = {
        'weighted_recent_form': 2.5,
        'speed_trend': 0.1,
        'speed_consistency': 0.7,
        'venue_win_rate': 0.2,
        'venue_avg_position': 3.5,
        'venue_experience': 8,
        'distance_win_rate': 0.18,
        'distance_avg_time': 30.1,
        'box_position_win_rate': 0.25,
        'box_position_avg': 3.0,
        'recent_momentum': 0.6,
        'competitive_level': 0.6,
        'position_consistency': 0.7,
        'top_3_rate': 0.4,
        'break_quality': 0.7
    }
    
    X = pd.DataFrame([test_features], columns=metadata.feature_names)
    print(f"\nOriginal features:")
    for feature, value in test_features.items():
        print(f"{feature:25s}: {value:8.4f}")
    
    # Apply scaling
    X_scaled = scaler.transform(X)
    print(f"\nScaled features:")
    for i, feature in enumerate(metadata.feature_names):
        print(f"{feature:25s}: {X_scaled[0][i]:8.4f}")
    
    # Check for extreme values
    extreme_values = []
    for i, feature in enumerate(metadata.feature_names):
        if abs(X_scaled[0][i]) > 5:  # Arbitrary threshold for "extreme"
            extreme_values.append((feature, X_scaled[0][i]))
    
    if extreme_values:
        print(f"\nExtreme scaled values (>5 or <-5):")
        for feature, value in extreme_values:
            print(f"{feature:25s}: {value:8.4f}")
    
    # Test prediction
    raw_pred = model.predict_proba(X)[0][1]
    scaled_pred = model.predict_proba(pd.DataFrame(X_scaled, columns=metadata.feature_names))[0][1]
    
    print(f"\nPredictions:")
    print(f"Raw prediction: {raw_pred:.6f}")
    print(f"Scaled prediction: {scaled_pred:.6f}")
    
    # Test with the training data range (if we can infer it)
    print(f"\nTesting with potential training range values:")
    
    # Try values closer to the scaler's center
    center_based_features = {}
    if hasattr(scaler, 'center_'):
        for i, feature in enumerate(metadata.feature_names):
            center_based_features[feature] = scaler.center_[i]
        
        X_center = pd.DataFrame([center_based_features], columns=metadata.feature_names)
        center_pred = model.predict_proba(X_center)[0][1]
        print(f"Center-based prediction: {center_pred:.6f}")
        
        print(f"\nCenter-based features:")
        for feature, value in center_based_features.items():
            print(f"{feature:25s}: {value:8.4f}")

else:
    print("No model found in registry")
