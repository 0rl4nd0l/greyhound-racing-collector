# Model Artifacts for Testing

This directory contains links to real model artifacts stored in cloud storage for testing purposes.

## Available Artifacts

### Lightweight Model (v1.0.0)
- **Size**: ~50MB 
- **Type**: RandomForest ensemble model
- **S3 Link**: `s3://greyhound-models-test/lightweight-model-v1.0.0.pkl`
- **GCS Link**: `gs://greyhound-models-test/lightweight-model-v1.0.0.pkl`
- **Local Download**: Use the script below to download for testing

```bash
# Download model artifact for testing
wget https://github.com/your-org/greyhound-models/releases/download/v1.0.0/lightweight-model-v1.0.0.pkl -O tests/artefacts/test-model.pkl
```

### Feature Scaler (v1.0.0)
- **Size**: ~1MB
- **Type**: StandardScaler for feature normalization
- **S3 Link**: `s3://greyhound-models-test/feature-scaler-v1.0.0.pkl`
- **GCS Link**: `gs://greyhound-models-test/feature-scaler-v1.0.0.pkl`

## Usage in Tests

```python
import joblib
import os

# Load test model
model_path = os.path.join("tests", "artefacts", "test-model.pkl")
if os.path.exists(model_path):
    test_model = joblib.load(model_path)
else:
    # Fallback to creating a minimal model for testing
    from sklearn.ensemble import RandomForestClassifier
    test_model = RandomForestClassifier(n_estimators=5, random_state=42)
```

## Data Compliance

- All models in this directory are **anonymized** and contain **no sensitive data**
- Models are trained on synthetic or publicly available datasets only
- Real production models are never stored in version control
