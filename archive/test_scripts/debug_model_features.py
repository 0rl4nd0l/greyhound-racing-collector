from model_registry import get_model_registry

# Get the global model registry
registry = get_model_registry()

# Get the best model
best_model_result = registry.get_best_model()

if best_model_result:
    model, scaler, metadata = best_model_result
    print(f"Model ID: {metadata.model_id}")
    print(f"Feature Names: {metadata.feature_names}")
    print(f"Model Type: {metadata.model_name}")
    print(f"Performance:")
    print(f"  Accuracy: {metadata.accuracy:.3f}")
    print(f"  AUC: {metadata.auc:.3f}")
    print(f"  F1: {metadata.f1_score:.3f}")
else:
    print("No model found in registry")
