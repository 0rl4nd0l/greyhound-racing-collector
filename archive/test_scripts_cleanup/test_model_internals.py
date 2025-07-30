from model_registry import get_model_registry
import numpy as np

# Get the global model registry
registry = get_model_registry()

# Get the best model
best_model_result = registry.get_best_model()

if best_model_result:
    model, scaler, metadata = best_model_result
    print(f"Model Info:")
    print(f"  ID: {metadata.model_id}")
    print(f"  Type: {metadata.model_name}")
    print(f"  Feature Names: {metadata.feature_names}")
    
    # Create test feature vector (random values)
    test_features = np.random.random(len(metadata.feature_names))
    X_test = test_features.reshape(1, -1)
    
    # Test prediction
    print("\nPrediction Test:")
    try:
        # Try basic prediction
        pred = model.predict(X_test)
        print(f"  Basic prediction: {pred}")
        
        # Try probability prediction
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_test)[0]
            print(f"  Probability prediction: {pred_proba}")
        
        # Try decision path if available
        if hasattr(model, 'decision_path'):
            decision_path = model.decision_path(X_test)
            print(f"  Decision path nodes: {decision_path.indices}")
            
        # Check model attributes
        print("\nModel Attributes:")
        print(f"  Feature count: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            print(f"  Classes: {model.classes_}")
        if hasattr(model, 'feature_importances_'):
            top_features = sorted(zip(metadata.feature_names, model.feature_importances_), 
                               key=lambda x: x[1], reverse=True)[:5]
            print("  Top features:")
            for feature, importance in top_features:
                print(f"    {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"Error testing model: {e}")
else:
    print("No model found in registry")
