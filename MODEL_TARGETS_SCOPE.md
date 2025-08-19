# Model Targets Scope & Entry Points Analysis

## Overview
Analysis of the three ML model targets for the greyhound racing prediction system, their implementation locations, import paths, and CLI invocation methods.

## Target Models Confirmed

### 1. "v3" = MLSystemV3 inside PredictionPipelineV3 (`prediction_method == ml_system_v3_basic`)

**Status**: ✅ **CONFIRMED** - Fully implemented and active

**Primary Implementation**:
- **File**: `ml_system_v3.py` (955 lines) - Main MLSystemV3 class
- **File**: `prediction_pipeline_v3.py` (565 lines) - PredictionPipelineV3 wrapper

**Import Path**:
```python
from ml_system_v3 import MLSystemV3
from prediction_pipeline_v3 import PredictionPipelineV3
```

**Entry Points**:
- **Direct instantiation**: `MLSystemV3(db_path="greyhound_racing_data.db")`
- **Through pipeline**: `PredictionPipelineV3(db_path="greyhound_racing_data.db")`
- **Training function**: `ml_system_v3.train_new_model(model_type="gradient_boosting")`
- **Web API**: Flask app endpoints with `prediction_method == "ml_system_v3_basic"`

**CLI Commands**:
- Train model: `python -c "from ml_system_v3 import train_new_model; train_new_model()"`
- No dedicated CLI script found

**Prediction Method Identifier**: `"ml_system_v3_basic"`

**Key Features**:
- Comprehensive integrated ML system
- Weather-enhanced predictions
- GPT analysis integration (optional)
- Drift monitoring with DriftMonitor
- Probability calibration
- SHAP explainability
- Champion/challenger model promotion
- Traditional analysis fallback

### 2. "v3s" = Standalone Simplified V3

**Status**: ⚠️ **PARTIALLY AVAILABLE** - Found standalone prediction script but not a full V3 system

**Implementation Found**:
- **File**: `outdated_scripts/prediction_agent/standalone_prediction.py` (303 lines)

**Import Path**:
```python
# Note: Located in outdated_scripts directory
from outdated_scripts.prediction_agent.standalone_prediction import run_comprehensive_prediction
```

**Entry Points**:
- **Direct execution**: `python outdated_scripts/prediction_agent/standalone_prediction.py`
- **Function call**: `run_comprehensive_prediction(race_file_path)`

**CLI Commands**:
- `python outdated_scripts/prediction_agent/standalone_prediction.py`

**Key Features**:
- Data-driven predictions using `ultra_insights.json`
- No OpenAI API required
- Weight bracket analysis
- Speed analysis
- Feature importance scoring
- Standalone operation (no ML training required)

**Note**: This is NOT a full MLSystemV3 implementation but rather a simplified standalone predictor. A true "V3s" (simplified V3) appears to be **an alias to the basic MLSystemV3 implementation** rather than a separate system.

**Recommendation**: The "v3s" target should be treated as an **alias to MLSystemV3** with simplified configuration or a subset of features disabled.

### 3. "v4" = MLSystemV4 (Current leakage-safe calibrated pipeline)

**Status**: ✅ **CONFIRMED** - Fully implemented and active

**Primary Implementation**:
- **File**: `ml_system_v4.py` (885 lines) - Main MLSystemV4 class  
- **File**: `prediction_pipeline_v4.py` - Pipeline wrapper
- **File**: `temporal_feature_builder.py` - Temporal leakage protection

**Import Path**:
```python
from ml_system_v4 import MLSystemV4
from prediction_pipeline_v4 import PredictionPipelineV4  # if exists
```

**Entry Points**:
- **Direct instantiation**: `MLSystemV4(db_path="greyhound_racing_data.db")`
- **Training function**: `ml_system_v4.train_leakage_safe_model()`
- **Backward compatibility**: `ml_system_v4.train_new_model()`

**CLI Commands**:
- **Training**: `python train_model_v4.py` (625 lines) - Conditional retraining manager
- **Direct training**: `python -c "from ml_system_v4 import train_leakage_safe_model; train_leakage_safe_model()"`
- **Test**: `python ml_system_v4.py` (direct execution)

**Key Features**:
- **Temporal leakage protection** via TemporalFeatureBuilder
- **Time-ordered train/test splits** (no random splits)
- **Proper sklearn pipeline** with ColumnTransformer
- **ExtraTreesClassifier with CalibratedClassifierCV**
- **Group-normalized probabilities** (softmax per race)
- **Expected Value (EV) calculation**
- **Comprehensive testing and validation**
- **Conditional retraining** with multiple triggers
- **Artifact management** with versioning

## Batch Processing CLI Options

### Available CLI Tools:
1. **`batch_predict_cli.py`** (529 lines) - Interactive CLI with progress tracking
2. **`batch_prediction_cli.py`** (452 lines) - Command-line batch processor  
3. **`cli_batch_predictor.py`** (366 lines) - Simplified batch CLI

### CLI Usage Examples:

#### Interactive Batch Processing:
```bash
python batch_predict_cli.py create
python batch_predict_cli.py run /path/to/csvs --output-dir ./results
python batch_predict_cli.py list
python batch_predict_cli.py monitor <job_id>
```

#### Command-line Batch Processing:
```bash
python batch_prediction_cli.py --input ./upcoming_races --output ./results
python batch_prediction_cli.py --input ./data --output ./results --workers 4 --batch-size 20
python batch_prediction_cli.py --job-status <job_id>
python batch_prediction_cli.py --list-jobs
```

#### Simple Batch Processing:
```bash
python cli_batch_predictor.py --file race.csv
python cli_batch_predictor.py --batch ./upcoming_races/
python cli_batch_predictor.py --upcoming-races
python cli_batch_predictor.py --list-jobs
```

## Model Selection in Prediction Pipeline

The model selection in `PredictionPipelineV3` follows this fallback hierarchy:

1. **Comprehensive Pipeline** (V3 Primary) - Full featured system
2. **Weather-Enhanced Predictor** (Fallback 1) - Weather integration
3. **Unified Predictor** (Fallback 2) - General purpose  
4. **Basic ML System V3** (Final Fallback) - `prediction_method == "ml_system_v3_basic"`

## Summary

| Target | Status | Primary File | Import Path | CLI Command |
|--------|--------|--------------|-------------|-------------|
| **v3** | ✅ Active | `ml_system_v3.py` | `from ml_system_v3 import MLSystemV3` | Manual training call |
| **v3s** | ⚠️ Alias | `ml_system_v3.py` | Same as v3 (simplified config) | Same as v3 |
| **v4** | ✅ Active | `ml_system_v4.py` | `from ml_system_v4 import MLSystemV4` | `python train_model_v4.py` |

## Recommendations

1. **v3s Target**: Implement as a configuration flag on MLSystemV3 rather than a separate system
2. **CLI Integration**: All three targets should be accessible through the batch processing CLIs
3. **Model Training**: Both v3 and v4 have comprehensive training capabilities
4. **Production Use**: v4 is recommended for production due to temporal leakage protection
