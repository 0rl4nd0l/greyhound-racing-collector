# Configuration Loader System

## Overview

The `config_loader.py` module provides a unified configuration management system for the Greyhound Racing Prediction System. It supports loading configuration from multiple sources with a clear priority hierarchy.

## Features

- **Multi-source Configuration**: YAML files, environment variables, and CLI flags
- **Priority System**: CLI flags > Environment variables > YAML configuration > Defaults
- **Type Safety**: Automatic type conversion and validation
- **Error Handling**: Comprehensive error reporting and validation
- **Integration Ready**: Easy integration with existing logging system
- **Fallback Support**: Works even when PyYAML is not available

## Supported Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence` | float | 0.20 | Minimum prediction confidence threshold |
| `max_calibration_error` | float | 0.15 | Maximum acceptable calibration error |
| `form_guide_required_fields` | int | 10 | Number of required fields in form guide |
| `drift_window_days` | int | 30 | Number of days for model drift detection window |
| `imbalance_ratio_warn` | float | 0.3 | Warning threshold for data imbalance ratio |

## Usage Examples

### Basic Usage

```python
from config_loader import get_config_loader, get_config

# Method 1: Get configuration dictionary
config = get_config()
min_conf = config['min_confidence']

# Method 2: Use config loader instance
loader = get_config_loader()
min_conf = loader.get_min_confidence()
```

### Environment Variables

Set environment variables to override default values:

```bash
export MIN_CONFIDENCE=0.25
export DRIFT_WINDOW_DAYS=45
export FORM_GUIDE_REQUIRED_FIELDS=15
python your_script.py
```

### CLI Flags

Use command-line flags for runtime configuration:

```bash
python your_script.py --min-confidence 0.3 --drift-window-days 21
```

### Custom Configuration File

```python
from config_loader import ConfigLoader

# Use a different config file
loader = ConfigLoader('config/production.yaml')
config = loader.get_config()
```

### Integration in Prediction Systems

```python
from config_loader import get_config_loader

def make_prediction(race_data):
    config = get_config_loader()
    
    # Use configuration parameters
    min_confidence = config.get_min_confidence()
    max_cal_error = config.get_max_calibration_error()
    
    # Your prediction logic here
    prediction_confidence = calculate_confidence(race_data)
    calibration_error = calculate_calibration_error(race_data)
    
    if prediction_confidence >= min_confidence and calibration_error <= max_cal_error:
        return {"prediction": "Winner", "confidence": prediction_confidence}
    else:
        return {"prediction": "Skip", "reason": "Below threshold"}
```

## YAML Configuration Structure

The system can extract configuration from the existing `config/defaults.yaml` structure:

```yaml
confidence_thresholds:
  minimum_confidence: 0.20  # Maps to min_confidence

model_registry:
  maximum_test_brier_score: 0.15  # Maps to max_calibration_error

drift_thresholds:
  performance_window_days: 30  # Maps to drift_window_days

data_quality:
  critical_fields_completeness: 0.95  # Maps to form_guide_required_fields
  maximum_duplicate_records_pct: 0.01  # Maps to imbalance_ratio_warn
```

## Environment Variable Mapping

| Parameter | Environment Variable |
|-----------|---------------------|
| `min_confidence` | `MIN_CONFIDENCE` |
| `max_calibration_error` | `MAX_CALIBRATION_ERROR` |
| `form_guide_required_fields` | `FORM_GUIDE_REQUIRED_FIELDS` |
| `drift_window_days` | `DRIFT_WINDOW_DAYS` |
| `imbalance_ratio_warn` | `IMBALANCE_RATIO_WARN` |

## CLI Flag Mapping

| Parameter | CLI Flag |
|-----------|----------|
| `min_confidence` | `--min-confidence` |
| `max_calibration_error` | `--max-calibration-error` |
| `form_guide_required_fields` | `--form-guide-required-fields` |
| `drift_window_days` | `--drift-window-days` |
| `imbalance_ratio_warn` | `--imbalance-ratio-warn` |

## Priority System

Configuration values are loaded in this order (later sources override earlier ones):

1. **Default Values**: Hard-coded defaults in the system
2. **YAML Configuration**: Values from the configuration file
3. **Environment Variables**: System environment variables
4. **CLI Flags**: Command-line arguments (highest priority)

Example demonstrating priority:
```bash
# YAML file has min_confidence: 0.20
# Environment variable: MIN_CONFIDENCE=0.25
# CLI flag: --min-confidence 0.30

# Result: min_confidence = 0.30 (CLI flag wins)
```

## Error Handling

The system provides comprehensive error handling:

- **File Not Found**: Falls back to defaults if YAML file doesn't exist
- **Invalid YAML**: Reports parsing errors with detailed messages
- **Type Errors**: Validates parameter types and provides clear error messages
- **Range Validation**: Ensures parameters are within valid ranges
- **Graceful Degradation**: Works without PyYAML dependency

## Testing

Run the built-in tests:

```bash
# Test basic functionality
python config_loader.py

# Test with environment variables
MIN_CONFIDENCE=0.25 python config_loader.py

# Test with CLI flags
python config_loader.py --min-confidence 0.35

# Test priority system
MIN_CONFIDENCE=0.25 python config_loader.py --min-confidence 0.4

# Test usage patterns
python test_config_usage.py
```

## Integration with Existing Systems

The config loader integrates seamlessly with the existing logging system:

```python
# Automatically uses existing logging configuration
from config.logging_config import log_config

# Configuration loading is logged with appropriate details
# Logs go to config.jsonl and main_workflow.jsonl
```

## Advanced Usage

### Reload Configuration

```python
loader = get_config_loader()
# ... configuration changes in environment/files ...
loader.reload()  # Reload from all sources
```

### Custom CLI Arguments (for testing)

```python
# For testing without affecting global CLI parsing
loader = ConfigLoader(
    config_file='test_config.yaml',
    parse_cli_args=False,
    cli_args=['--min-confidence', '0.5']
)
```

### Non-Interactive Usage

```python
# Disable CLI argument parsing for library usage
loader = ConfigLoader(
    config_file='config/defaults.yaml',
    parse_cli_args=False
)
```

## Dependencies

- **Core**: Python 3.7+ (uses standard library)
- **Optional**: PyYAML (for YAML file support - falls back gracefully if not available)
- **Integration**: config.logging_config (falls back to standard logging if not available)

## Installation

The configuration loader is ready to use as-is. For full YAML support, ensure PyYAML is installed:

```bash
pip install PyYAML
```

## Best Practices

1. **Use Global Instance**: Use `get_config_loader()` for consistent configuration across your application
2. **Parameter Validation**: Always validate configuration parameters in your application logic
3. **Environment-Specific Config**: Use different YAML files for different environments (dev, staging, production)
4. **Documentation**: Document any new configuration parameters you add
5. **Testing**: Test configuration changes with different sources (YAML, env vars, CLI flags)

## Examples in Action

See `test_config_usage.py` for comprehensive examples of how to integrate the configuration loader into your applications.
