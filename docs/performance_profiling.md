# Performance Profiling Documentation

## Usage Examples

### Basic Profiling Execution
To profile your application's performance, follow the steps below:

1. **Enable Profiling:**
   ```bash
   export PROFILING_ENABLED=true
   ```

2. **Run the Profiler**
   ```bash
   python performance_profiler.py
   ```

3. **Viewing Results:**
   Profiles are saved in the `profiles` directory.
   
### Advanced Usage with py-spy

- **Running py-spy:**
   ```bash
   py-spy record --pid <FLASK_PID> --duration 30 --output profiles/py_spy_report.svg --format svg
   ```

- **Visualize:** Open the SVG file with your browser for detailed insights.

## Profiling Flag Description

- **`--profiling` Flag**
  - Enables/disables profiling during execution.
  - Usage: Include in CLI commands like `python script.py --profiling`

## Drift Report Interpretation

### Understanding Drift Results
Drift detection helps in identifying deviations in input datasets between training and prediction stages.

- **High Drift Features:** Automatically detected features with significant drift.
- **Drift Triggers:** Includes changes in mean, standard deviation, or PSI exceeding thresholds (0.25 for PSI).
- **Using Evidently and Manual Checks:** Manual checks with PSI are supported for columns not supported by Evidently.

### Taking Action
- Perform retraining if high drift is detected.
- Reports are saved in the `audit_results` directory.

## Upgrade Notes

### Updating from Previous Versions

- **Flag Changes:** Ensure you are using the `--profiling` flag in your CLI for consistent profiling across modules.
- **Drift Monitor:** Now includes both Evidently and Manual Drift Detection; ensure Evidently is installed for full feature access.

## Additional Notes

- Performance and drift profiling are crucial for maintaining high model accuracy and efficiency.
- Ensure directories like `profiles` and `audit_results` are writable by your application.
