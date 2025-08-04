# Greyhound Predictor Pipeline Bottleneck Analysis
Generated: 2025-08-04T00:58:12.527493

## Executive Summary
- **Functions Profiled**: 38
- **SQL Queries Analyzed**: 0
- **Slow SQL Queries**: 0
- **Sequence Steps Tracked**: 76

## Critical Bottlenecks
🔥 **_try_load_latest_model**: 5.252s (CPU: 5.156s)
🔥 **_try_load_latest_model**: 4.907s (CPU: 4.841s)
🔥 **_try_load_latest_model**: 6.950s (CPU: 7.065s)
🔥 **_try_load_latest_model**: 7.011s (CPU: 6.629s)
🔥 **_try_load_latest_model**: 13.114s (CPU: 13.411s)

## Optimization Recommendations
### CPU Optimization
- Optimize **_try_load_latest_model** - CPU bound (5.156s)
- Optimize **_try_load_latest_model** - CPU bound (4.841s)
- Optimize **_try_load_latest_model** - CPU bound (7.065s)
- Optimize **_try_load_latest_model** - CPU bound (6.629s)
- Optimize **_try_load_latest_model** - CPU bound (13.411s)

## Detailed Profiling Data
### Function Profiles
- **_try_load_latest_model**:
  - Total Time: 13.114s
  - CPU Time: 13.411s
  - I/O Time: 0.0ms
  - Memory Peak: 26.11MB
  - Function Calls: 4810729
  - Bottlenecks: <20>(2034.000s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:438(load_build)(70.459s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:259(read)(54.161s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:159(read_array)(52.574s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle_utils.py:234(_read_bytes)(40.485s)

- **_try_load_latest_model**:
  - Total Time: 7.011s
  - CPU Time: 6.629s
  - I/O Time: 0.0ms
  - Memory Peak: 35.07MB
  - Function Calls: 2429771
  - Bottlenecks: <20>(1051.000s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/threading.py:998(_bootstrap)(2.333s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py:493(send_response)(2.297s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/werkzeug/serving.py:259(write)(2.181s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py:513(shutdown_request)(2.015s)

- **_try_load_latest_model**:
  - Total Time: 6.950s
  - CPU Time: 7.065s
  - I/O Time: 0.0ms
  - Memory Peak: 25.54MB
  - Function Calls: 2702907
  - Bottlenecks: <20>(548.000s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:438(load_build)(5.159s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:259(read)(3.313s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/threading.py:998(_bootstrap)(3.070s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/threading.py:1025(_bootstrap_inner)(3.070s)

- **_try_load_latest_model**:
  - Total Time: 5.252s
  - CPU Time: 5.156s
  - I/O Time: 0.0ms
  - Memory Peak: 25.71MB
  - Function Calls: 2332301
  - Bottlenecks: <20>(285.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:629(_try_load_latest_model)(5.249s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:674(load)(5.232s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:613(_unpickle)(5.211s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/pickle.py:1230(load)(4.738s)

- **_try_load_latest_model**:
  - Total Time: 4.907s
  - CPU Time: 4.841s
  - I/O Time: 0.0ms
  - Memory Peak: 25.24MB
  - Function Calls: 2331783
  - Bottlenecks: <20>(234.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:629(_try_load_latest_model)(4.906s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:674(load)(4.903s), /Users/orlandolee/greyhound_racing_collector/venv/lib/python3.13/site-packages/joblib/numpy_pickle.py:613(_unpickle)(4.884s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/pickle.py:1230(load)(4.385s)

- **_group_normalize_probabilities**:
  - Total Time: 0.005s
  - CPU Time: 0.004s
  - I/O Time: 0.0ms
  - Memory Peak: 0.13MB
  - Function Calls: 181
  - Bottlenecks: <20>(117.000s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/logging/__init__.py:1538(error)(0.002s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/logging/__init__.py:1640(_log)(0.002s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/logging/__init__.py:1666(handle)(0.001s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/logging/__init__.py:1720(callHandlers)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.004s
  - CPU Time: 0.002s
  - I/O Time: 0.0ms
  - Memory Peak: 0.03MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.003s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:698(track_sequence)(0.002s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:704(__init__)(0.001s), objects}(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.003s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.002s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:714(__exit__)(0.002s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:307(track_sequence_step)(0.002s)

- **_group_normalize_probabilities**:
  - Total Time: 0.002s
  - CPU Time: 0.002s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.002s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:714(__exit__)(0.001s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:307(track_sequence_step)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.002s
  - CPU Time: 0.002s
  - I/O Time: 0.0ms
  - Memory Peak: 0.03MB
  - Function Calls: 36
  - Bottlenecks: <20>(32.000s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/connection.py:131(__del__)(0.001s), /usr/local/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/connection.py:376(_close)(0.001s), posix.close}(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.002s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:698(track_sequence)(0.001s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:704(__init__)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.002s
  - CPU Time: 0.002s
  - I/O Time: 0.0ms
  - Memory Peak: 0.13MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:714(__exit__)(0.001s), /Users/orlandolee/greyhound_racing_collector/pipeline_profiler.py:307(track_sequence_step)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.002s
  - CPU Time: 0.002s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s), /Users/orlandolee/greyhound_racing_collector/ml_system_v4.py:470(_group_normalize_probabilities)(0.001s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.02MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)

- **_group_normalize_probabilities**:
  - Total Time: 0.001s
  - CPU Time: 0.001s
  - I/O Time: 0.0ms
  - Memory Peak: 0.01MB
  - Function Calls: 27
  - Bottlenecks: <20>(23.000s)
