# Greyhound Predictor Pipeline Bottleneck Analysis
Generated: 2025-08-04T15:18:54.284862

## Executive Summary
- **Functions Profiled**: 2
- **SQL Queries Analyzed**: 0
- **Slow SQL Queries**: 0
- **Sequence Steps Tracked**: 3

## Critical Bottlenecks
🔥 **_try_load_latest_model**: 33.110s (CPU: 27.597s)

## Optimization Recommendations
### CPU Optimization
- Optimize **_try_load_latest_model** - CPU bound (27.597s)
### Memory Optimization
- Optimize **_try_load_latest_model** - High memory usage (295.23MB)

## Detailed Profiling Data
### Function Profiles
- **_try_load_latest_model**:
  - Total Time: 33.110s
  - CPU Time: 27.597s
  - I/O Time: 0.0ms
  - Memory Peak: 295.23MB
  - Function Calls: 7734050
  - Bottlenecks: <20>(281.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:984(_try_load_latest_model)(33.108s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:674(load)(33.081s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:613(_unpickle)(32.822s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/pickle.py:1230(load)(32.822s)

- **_group_normalize_probabilities**:
  - Total Time: 0.006s
  - CPU Time: 0.005s
  - I/O Time: 0.0ms
  - Memory Peak: 0.04MB
  - Function Calls: 65
  - Bottlenecks: <20>(35.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:791(_group_normalize_probabilities)(0.006s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:714(__exit__)(0.002s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:307(track_sequence_step)(0.002s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:4073(var)(0.001s)
