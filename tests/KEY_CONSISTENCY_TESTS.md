# Key Consistency Regression & Unit Tests - Step 6

## Overview

This test suite implements **Step 6** of the Greyhound Analysis Predictor improvement plan: comprehensive regression and unit tests for key consistency across all prediction layers.

## Purpose

The key consistency tests ensure that:

1. **No KeyErrors occur** when loading `tests/fixtures/test_race.csv` through prediction layers
2. **Prediction tiers never fall back to `"dummy_fallback"`** unexpectedly  
3. **All loaders accept constant keys** consistently
4. **Error handling is robust** and doesn't crash with KeyErrors
5. **CI pipeline blocks merges** when key consistency regressions are detected

## Test Structure

### Files

- `tests/test_key_consistency.py` - Main test suite
- `tests/run_key_consistency_tests.py` - Comprehensive test runner
- `tests/run_ci_key_consistency.py` - CI-optimized test runner
- `tests/fixtures/test_race.csv` - Test data fixture
- `tests/KEY_CONSISTENCY_TESTS.md` - This documentation

### Test Categories

#### 1. Basic CSV Structure Tests (`@pytest.mark.unit`)
- **`test_test_race_csv_exists`** - Verifies test fixture exists and is readable
- **`test_csv_structure_and_key_consistency`** - Validates CSV structure and key mappings

#### 2. Parametrized Layer Tests (`@pytest.mark.integration`)
- **`test_prediction_layers_key_consistency`** - Tests ML, weather, and unified predictors
- **`test_loaders_accept_constant_keys`** - Validates CSV loaders accept constant keys

#### 3. Pipeline Integration Tests (`@pytest.mark.integration`)
- **`test_prediction_pipeline_no_key_errors`** - Tests main pipeline for KeyErrors
- **`test_weather_enhanced_predictor_key_handling`** - Weather predictor key handling

#### 4. Error Handling Tests (`@pytest.mark.integration`) 
- **`test_error_handling_and_fallback_logic`** - Validates fallback logic doesn't crash

#### 5. Full Integration Tests (`@pytest.mark.slow`)
- **`test_integration_all_layers_consistent_keys`** - End-to-end layer consistency

## Test Markers

```python
@pytest.mark.key_consistency  # All key consistency tests
@pytest.mark.unit            # Fast unit tests
@pytest.mark.integration     # Integration tests requiring components
@pytest.mark.slow            # Slow tests (excluded from CI fast runs)
```

## Running Tests

### Local Development

```bash
# Run all key consistency tests
pytest tests/test_key_consistency.py -m key_consistency -v

# Run only fast tests (recommended)
pytest tests/test_key_consistency.py -m "key_consistency and not slow" -v

# Run with comprehensive reporting
python tests/run_key_consistency_tests.py
```

### CI Environment

```bash
# CI-optimized runner (used in GitHub Actions)
python tests/run_ci_key_consistency.py
```

## CI Integration

### GitHub Actions Integration

The tests are integrated into both CI workflows:

1. **`.github/workflows/backend-tests.yml`** - Backend test matrix
2. **`.github/workflows/ci.yml`** - Main CI pipeline

Both workflows run the fast tests only to prevent CI timeouts:

```yaml
- name: Run key consistency regression tests
  run: |
    python -m pytest tests/test_key_consistency.py \
      -m "key_consistency and not slow" \
      --tb=short \
      -v \
      --maxfail=3 \
      --timeout=300
  continue-on-error: false  # Blocks merges on failure
```

### Failure Scenarios

The CI will **block merges** when:

- Any KeyErrors are detected in prediction layers
- Prediction tiers unexpectedly fall back to `"dummy_fallback"`
- CSV loaders fail to accept constant keys
- Tests timeout (indicating performance regression)

## Test Fixtures

### `tests/fixtures/test_race.csv`

Contains realistic race data with 8 dogs across 4 historical races:

```csv
Dog Name,BOX,WGT,TIME,SP,PLC,MGN,DIST,G,TRACK,DATE,Sex,1 SEC
1. SUPER FAST,1,32.5,30.20,2.40,1,0.5,500,5,GEE,2025-01-15,D,4.85
2. LIGHTNING BOLT,2,31.8,30.15,3.50,2,0.3,500,5,GEE,2025-01-15,D,4.87
...
```

## Constant Keys

Tests validate that these constant keys are handled consistently:

```python
from constants import DOG_NAME_KEY, DOG_BOX_KEY, DOG_WEIGHT_KEY

# Expected mappings
CSV_COLUMN -> CONSTANT_KEY
'Dog Name' -> DOG_NAME_KEY  ("dog_name")
'BOX'      -> DOG_BOX_KEY   ("box")  
'WGT'      -> DOG_WEIGHT_KEY ("weight")
```

## Expected Results

### Successful Run
```
🔧 Key Consistency Test Runner - Step 6 Implementation
============================================================
📋 Running Basic CSV Structure Tests... ✅ (2/2 passed)
📋 Running Parametrized Layer Tests... ✅ (2/2 passed)  
📋 Running Pipeline Integration Tests... ✅ (2/2 passed)
📋 Running Error Handling & Fallback Tests... ✅ (1/1 passed)
📋 Running Full Integration Tests... ✅ (1/1 passed)

🏁 Key Consistency Test Summary
============================================================
Total Tests: 8
Passed: ✅ 8
Failed: ❌ 0
KeyErrors Detected: 🚨 0

🎉 All key consistency tests passed!
✅ No KeyErrors detected - prediction layers are handling keys consistently
```

### Failed Run (Blocks CI)
```
🚨 CRITICAL: 2 KeyErrors detected!
This indicates regression in key handling that must be fixed before merge.
❌ 3 tests failed
```

## Performance

- **Fast tests** (~5 minutes): Unit + integration tests without slow components
- **Full tests** (~30+ minutes): Includes comprehensive pipeline initialization
- **CI timeout**: 5 minutes per test, 30 minutes total

## Troubleshooting

### Common Issues

1. **Test fixture missing**: Ensure `tests/fixtures/test_race.csv` exists
2. **Import errors**: Check that prediction modules can be imported
3. **Database issues**: Tests use temporary databases for isolation
4. **Timeout errors**: Use fast tests (`not slow`) for development

### Debug Commands

```bash
# Run specific test with full output
pytest tests/test_key_consistency.py::test_csv_structure_and_key_consistency -v -s

# Run with maximum detail
pytest tests/test_key_consistency.py -v -s --tb=long

# Check test collection
pytest tests/test_key_consistency.py --collect-only
```

## Contributing

When adding new prediction layers or modifying key handling:

1. **Add tests** for new components in `test_prediction_layers_key_consistency`
2. **Update fixtures** if new CSV structure is required
3. **Test locally** with `python tests/run_key_consistency_tests.py`
4. **Verify CI passes** fast tests before pushing

## Related Documentation

- [Greyhound Racing Predictor Architecture](../README.md)
- [Constants Documentation](../constants.py)
- [Prediction Pipeline Documentation](../prediction_pipeline_v3.py)
- [CI/CD Documentation](../.github/workflows/)

---

**Step 6 Complete**: Key consistency regression and unit tests implemented with CI integration to prevent KeyError regressions and ensure prediction layer reliability.
