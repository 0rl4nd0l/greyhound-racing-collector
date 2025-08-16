# SHAP Explainability Integration

**Objective:** Add SHAP explainability to provide insights into the model predictions of the Greyhound Analysis Predictor.

### Steps Implemented:
1. **SHAP Module Creation**
   - Created `shap_explainer.py` to automatically build a SHAP explainer for each base model and cache them using Joblib.
   - Supports both `TreeExplainer` for tree-based models and `KernelExplainer` for others.

2. **Integration with Prediction System**
   - Modified the ML system (`ml_system_v3.py`) to include SHAP explainability in predictions.
   - Enhanced the prediction pipeline (`prediction_pipeline_v3.py`) to embed SHAP explanations in each dog's output.
   - Added utility functions to easily fetch SHAP values and add them to the predictions.

3. **Testing and Validation**
   - Developed a test script (`tests/test_shap_integration.py`) to validate SHAP integration.
   - Tests ensure the correct addition of explainability insights into outputs.

### Usage:
- Retrieve explainability insights for specific predictions by analyzing the `explainability` field in the prediction JSON output.
- Top-N feature impacts can be retrieved for each prediction.

### Requirements:
- Ensure the SHAP library is installed (`pip install shap`).
- The ML models should be trained and available in the `model_registry/models/` directory.

