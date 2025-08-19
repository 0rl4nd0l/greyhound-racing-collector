# Model Registry

The Model Registry system provides centralized management for trained machine learning models, ensuring optimal model selection and deployment.

## Core Features

### Model Versioning
- **Unique Model IDs**: Each model receives a unique identifier based on name, type, and timestamp
- **Semantic Versioning**: Models follow semantic versioning patterns for clear version tracking
- **Metadata Tracking**: Comprehensive metadata stored for each model version

### Performance Tracking
```python
@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    model_type: str
    training_timestamp: str
    accuracy: float
    auc: float
    f1_score: float
    precision: float
    recall: float
    training_samples: int
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    is_active: bool = False
    is_best: bool = False
```

### Automatic Model Selection
The registry automatically selects the best performing model based on composite scoring:

```python
def _calculate_model_score(self, metadata: ModelMetadata) -> float:
    """Calculate composite model score for ranking"""
    weights = {
        'accuracy': 0.4,
        'auc': 0.3, 
        'f1_score': 0.2,
        'data_quality': 0.1
    }
    
    score = (
        metadata.accuracy * weights['accuracy'] +
        metadata.auc * weights['auc'] +
        metadata.f1_score * weights['f1_score'] +
        metadata.data_quality_score * weights['data_quality']
    )
    
    # Bonus for ensemble models
    if metadata.is_ensemble:
        score *= 1.05
    
    return score
```

## Model Lifecycle Management

### Registration Process
1. **Model Training**: Train new model with comprehensive metrics
2. **Performance Validation**: Validate model performance against benchmarks
3. **Registration**: Register model with metadata and artifacts
4. **Automatic Evaluation**: System evaluates model against current champion
5. **Promotion**: Best performing model becomes active

### Model Storage Structure
```
model_registry/
├── models/
│   ├── model_RandomForest_20250731_120000.pkl
│   ├── scaler_RandomForest_20250731_120000.pkl
│   └── ...
├── metadata/
│   ├── model_RandomForest_20250731_120000.json
│   └── ...
├── model_index.json
└── registry_config.json
```

### Model Artifacts
- **Model Files**: Serialized model objects (pickle/joblib format)
- **Scaler Files**: Feature scaling objects for preprocessing
- **Metadata**: Comprehensive model information and performance metrics
- **Configuration**: Registry settings and performance weights

## Performance Monitoring

### Model Comparison
```python
def get_model_comparison(self, limit: int = 10) -> List[Dict[str, Any]]:
    """Get comparison of top performing models"""
    models_with_scores = []
    
    for model_id, metadata_dict in self.model_index.items():
        metadata = ModelMetadata(**metadata_dict)
        score = self._calculate_model_score(metadata)
        
        models_with_scores.append({
            'model_id': model_id,
            'model_name': metadata.model_name,
            'model_type': metadata.model_type,
            'accuracy': metadata.accuracy,
            'auc': metadata.auc,
            'f1_score': metadata.f1_score,
            'composite_score': score,
            'is_active': metadata.is_active,
            'training_date': metadata.training_timestamp
        })
    
    # Sort by composite score (descending)
    models_with_scores.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return models_with_scores[:limit]
```

### Performance Degradation Detection
- **Continuous Monitoring**: Track model performance over time
- **Threshold Alerting**: Alert when performance drops below thresholds
- **Automatic Rollback**: Revert to previous best model if current model fails

## Model Deployment

### Champion/Challenger System
- **Champion Model**: Currently deployed production model
- **Challenger Models**: New models being evaluated
- **A/B Testing**: Gradual rollout of challenger models
- **Performance Comparison**: Continuous comparison between champion and challengers

### Model Loading
```python
def load_best_model(self) -> Tuple[Any, Any, ModelMetadata]:
    """Load the best performing model for predictions"""
    best_model_id = self.get_best_model_id()
    
    if not best_model_id:
        raise ValueError("No models available in registry")
    
    metadata = ModelMetadata(**self.model_index[best_model_id])
    
    # Load model and scaler
    model = joblib.load(metadata.model_file_path)
    scaler = joblib.load(metadata.scaler_file_path)
    
    return model, scaler, metadata
```

## Integration with ML Pipeline

### Training Integration
The registry integrates seamlessly with the ML training pipeline:

```python
# Train model
model, scaler, performance_metrics = train_model(training_data)

# Register in registry
model_id = registry.register_model(
    model_obj=model,
    scaler_obj=scaler,
    model_name="GreyhoundPredictor",
    model_type="RandomForest",
    performance_metrics=performance_metrics,
    training_info=training_info,
    feature_names=feature_names,
    hyperparameters=hyperparameters
)

# Auto-select best model
best_model_id = registry.get_best_model_id()
```

### Prediction Integration
Prediction systems automatically use the best available model:

```python
def make_prediction(self, features):
    # Load best model from registry
    model, scaler, metadata = self.registry.load_best_model()
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict_proba(scaled_features)
    
    return prediction, metadata.model_id
```

## Configuration and Maintenance

### Registry Configuration
```json
{
  "auto_select_best": true,
  "max_models_to_keep": 50,
  "performance_weight": {
    "accuracy": 0.4,
    "auc": 0.3,
    "f1_score": 0.2,
    "data_quality": 0.1
  }
}
```

### Cleanup and Maintenance
- **Automatic Cleanup**: Remove old models based on retention policy
- **Storage Optimization**: Compress old model artifacts
- **Performance Archiving**: Archive performance metrics for historical analysis
- **Index Rebuilding**: Periodic index rebuilding for optimal performance

The Model Registry ensures reliable, high-performance model management with automatic optimization and robust deployment capabilities.
