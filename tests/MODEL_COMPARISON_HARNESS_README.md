# Model Comparison Harness

A unified test harness for comparing ML models (v3, v3s, v4, all) against CSV race data with comprehensive preprocessing and analysis capabilities.

## Overview

The Model Comparison Harness implements **Step 2** of the broader testing plan by providing a standardized framework for evaluating multiple ML models against the same race data with proper temporal leakage prevention.

## Features

### 🔧 Core Functionality
- **Multi-model Support**: Tests v3, v3s, v4, or all models simultaneously
- **CSV Ingestion**: Robust CSV loading with `CsvIngestion` integration
- **FORM_GUIDE_SPEC.md Compliance**: Applies standardized preprocessing rules
- **Temporal Leakage Prevention**: Automatically strips post-outcome columns
- **Standardized Output**: Collects raw/normalized probabilities, ranks, and metadata

### 📊 Data Processing
- **Column Mapping**: Maps "Dog Name" to `dog_name` consistently
- **Blank Row Handling**: Forward-fills dog names for continuation rows
- **Post-Outcome Stripping**: Removes finish positions, times, margins, winners
- **Default Value Injection**: Adds missing required fields with sensible defaults
- **Validation**: Ensures data quality before model injection

### 🤖 Model Integration
- **V3 Integration**: Full ML system with traditional analysis
- **V3S Integration**: Simplified version (disabled advanced features)
- **V4 Integration**: Temporal leakage-safe with calibration & EV
- **Mock Predictions**: Generates intelligent mock predictions when models aren't trained
- **Error Handling**: Graceful fallbacks for model failures

### 📈 Analysis & Reporting
- **Race Summaries**: Real-time top pick comparisons
- **JSON Output**: Detailed results with full metadata
- **Statistics**: Success rates, confidence scores, calibration usage
- **Comparison Metrics**: Cross-model performance analysis

## Usage

### Basic Usage

```bash
# Test single model
python tests/model_comparison_harness.py --model v3 --csv-dir data/test_races/

# Test all models
python tests/model_comparison_harness.py --model all --csv-dir data/upcoming_races/

# Verbose logging
python tests/model_comparison_harness.py --model v4 --csv-dir data/validation/ --verbose
```

### Command Line Options

```
--model {v3|v3s|v4|all}     Model(s) to test
--csv-dir PATH              Directory containing CSV files  
--db-path PATH              SQLite database path (optional)
--verbose                   Enable debug logging
```

### CSV Format Requirements

The harness expects CSV files with the following columns (case-insensitive):

#### Required Columns
- `Dog Name` / `DOG NAME` / `Name` → `dog_name`
- `BOX` / `Box` → `box_number`
- `PIR` → `pir_rating` (Performance Index Rating)
- `SP` → `starting_price`

#### Optional Columns  
- `WGT` / `Weight` → `weight`
- `DIST` / `Distance` → `distance`
- `TRACK` / `Venue` → `venue`
- `G` / `Grade` → `grade`

#### Post-Outcome Columns (Automatically Stripped)
- `PLC` / `Place` / `Position` → `finish_position` ❌
- `TIME` / `Individual_Time` → `individual_time` ❌
- `MGN` / `Margin` → `margin` ❌
- `Winner_Name` / `First_Place` → `winner_name` ❌

## Model Descriptions

### V3 Model
- **Type**: Full ML System with Traditional Analysis
- **Features**: Weather integration, GPT analysis, comprehensive feature engineering
- **Normalization**: Individual dog predictions
- **Calibration**: Available if trained model exists
- **Fallback**: Mock predictions based on PIR and starting price

### V3S Model (Simplified)
- **Type**: Basic ML System
- **Features**: Core ML without drift monitoring or traditional analysis
- **Normalization**: Individual dog predictions
- **Calibration**: Available if trained model exists
- **Fallback**: Same mock prediction logic as V3

### V4 Model
- **Type**: Temporal Leakage-Safe with Calibration
- **Features**: ExtraTreesClassifier, isotonic calibration, EV calculation
- **Normalization**: Group softmax (probabilities sum to 1 per race)
- **Calibration**: Built-in CalibratedClassifierCV
- **Temporal Safety**: Strict temporal feature building and validation

## Output Format

### Race Summary (Console)
```
📊 Race Summary: test_race_1 (8 dogs)
============================================================
  V3: Top pick = Speedy Susan (Box 3, Win: 0.900, Rank: 1)
  V4: Top pick = Lightning Bolt (Box 1, Win: 0.342, Rank: 1)
```

### JSON Results Structure
```json
{
  "metadata": {
    "total_races": 3,
    "models_tested": ["v3", "v4"],
    "harness_version": "1.0.0",
    "timestamp": "2025-08-04T12:08:12.164423"
  },
  "results": [
    {
      "race_name": "test_race_1",
      "num_dogs": 8,
      "models": {
        "v3": {
          "success": true,
          "predictions": [
            {
              "dog_name": "Speedy Susan",
              "box_number": 3,
              "raw_win_probability": 0.850,
              "normalized_win_probability": 0.850,
              "predicted_rank": 1,
              "confidence": 0.75,
              "model_metadata": {
                "model_info": "gradient_boosting",
                "calibration_applied": true
              }
            }
          ],
          "model_metadata": {
            "model_type": "v3",
            "normalization_method": "individual",
            "has_calibration": true
          }
        }
      }
    }
  ]
}
```

## FORM_GUIDE_SPEC.md Integration

The harness implements all preprocessing rules from `FORM_GUIDE_SPEC.md`:

### ✅ Implemented Features

1. **Column Mapping**: Standardizes column names across different CSV formats
2. **Blank Row Handling**: Forward-fills dog names for continuation rows
3. **Mixed Delimiter Detection**: Handles inconsistent separators
4. **Unicode Cleanup**: Removes BOM and invisible characters
5. **Temporal Leakage Prevention**: Strips all post-outcome data
6. **Data Validation**: Ensures required fields and data quality
7. **Default Value Injection**: Adds missing race metadata

### Example Preprocessing

**Input CSV:**
```csv
Dog Name,BOX,WGT,PIR,SP,PLC,TIME
Lightning Bolt,1,32.5,78,4.50,2,29.85
Thunder Strike,2,33.1,72,6.20,3,30.12
```

**After Preprocessing:**
```csv
dog_name,box_number,weight,pir_rating,starting_price,venue,grade
Lightning Bolt,1,32.5,78,4.50,UNKNOWN,5
Thunder Strike,2,33.1,72,6.20,UNKNOWN,5
```

## Mock Predictions

When models don't have trained pipelines, the harness generates intelligent mock predictions:

### Mock Algorithm
```python
# Base probability from PIR (Performance Index Rating)
mock_win_prob = max(0.05, min(0.80, pir_rating / 100.0))

# Adjust based on starting price (lower price = higher probability)
if starting_price:
    price_factor = max(0.1, min(1.0, 10.0 / starting_price))
    mock_win_prob = (mock_win_prob + price_factor) / 2
```

### Mock Features
- **PIR-based**: Uses performance ratings as primary factor
- **Price-adjusted**: Incorporates betting market sentiment
- **Realistic Range**: Probabilities between 0.05 and 0.80
- **Low Confidence**: Always reports 0.3 confidence for mocks
- **Clear Flagging**: Sets `is_mock: true` in metadata

## Testing & Validation

### Demo Script
Run the included demo to verify functionality:

```bash
python test_harness_demo.py
```

### Test CSV Files
Sample CSV files provided in `tests/sample_csv_data/`:
- `test_race_1.csv` - Basic race data
- `test_race_2.csv` - Different venue/grade
- `test_race_with_outcomes.csv` - Includes post-outcome columns for leakage testing

### Expected Output
- ✅ CSV preprocessing with column mapping
- ✅ Post-outcome column stripping
- ✅ Mock prediction generation
- ✅ Race summary reporting
- ✅ JSON result export

## Error Handling

The harness provides comprehensive error handling:

### Model Loading Errors
- **Missing Dependencies**: Gracefully handles missing imports
- **Database Issues**: Continues with available models
- **Training Failures**: Falls back to mock predictions

### CSV Processing Errors
- **Invalid Format**: Skips problematic files with warnings
- **Missing Columns**: Uses sensible defaults where possible
- **Encoding Issues**: Attempts multiple encodings

### Prediction Errors
- **Model Failures**: Records errors in results
- **Data Issues**: Continues with available predictions
- **Network Timeouts**: Handles external service failures

## Performance Considerations

### Model Loading
- **V3/V3S**: May be slow due to comprehensive data loading
- **V4**: Faster initialization with temporal feature building
- **Mock Mode**: Instant predictions for testing

### Memory Usage
- **CSV Caching**: Loads all CSVs into memory for processing
- **Model Storage**: Keeps all loaded models in memory
- **Result Collection**: Stores all predictions for final report

### Optimization Tips
- Use `--model v4` for fastest testing
- Limit CSV directory size for large datasets
- Enable `--verbose` only for debugging

## Integration Points

### Existing Systems
- **CsvIngestion**: Uses existing CSV processing infrastructure
- **ML Models**: Integrates with v3, v4 prediction systems
- **FORM_GUIDE_SPEC**: Implements standardized preprocessing

### Future Extensions
- **Backtesting**: Could integrate with historical race results
- **Live Prediction**: Real-time race evaluation capabilities
- **Model Training**: Could trigger retraining based on performance
- **API Integration**: RESTful endpoints for external systems

## Troubleshooting

### Common Issues

**"No CSV files found"**
- Check directory path and file extensions
- Ensure files have `.csv` extension

**"Model loading timeout"**
- V3 models may take time to initialize
- Try V4 model first for faster testing
- Check database connectivity

**"No predictions generated"**
- Verify CSV column names match expected format
- Check for required columns (dog_name, box_number)
- Review preprocessing logs for errors

**"Mock predictions only"**
- Models need trained pipelines for real predictions
- Mock predictions still demonstrate harness functionality
- Check model training status

### Debug Mode
Enable verbose logging for detailed troubleshooting:

```bash
python tests/model_comparison_harness.py --model v3 --csv-dir data/ --verbose
```

## Contributing

### Adding New Models
1. Import model class in harness
2. Add model choice to argument parser
3. Implement model-specific prediction method
4. Update documentation

### Extending Preprocessing
1. Add new column mappings to `column_mapping` dict
2. Update post-outcome column list if needed
3. Add validation rules for new fields
4. Test with sample CSV files

### Improving Mock Predictions
1. Enhance mock algorithm in `get_v3_predictions`
2. Add more sophisticated probability calculations
3. Include additional race factors
4. Validate against known good predictions

## License & Credits

This harness implements the requirements from Step 2 of the broader ML model comparison plan, with full integration of FORM_GUIDE_SPEC.md preprocessing rules and comprehensive temporal leakage prevention.
