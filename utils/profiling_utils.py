import threading
import time
import json
import os
from datetime import datetime

from profiling_config import is_profiling

class ProfilingRecorder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProfilingRecorder, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.recordings = []
        self._data_lock = threading.Lock()

    def record(self, stage_name, duration):
        with self._data_lock:
            self.recordings.append({'stage_name': stage_name, 'duration': duration})

    def flush_to_file(self):
        with self._data_lock:
            if not self.recordings:
                return

            timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
            dir_name = f"audit_results/{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            file_path = os.path.join(dir_name, "perf_metrics.json")

            with open(file_path, 'w') as f:
                json.dump(self.recordings, f, indent=2)

            self.recordings = []

ProfilingRecorder = ProfilingRecorder()  # Singleton instance


def timed(stage_name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if not is_profiling():
                return fn(*args, **kwargs)

            start = time.time()
            result = fn(*args, **kwargs)
            duration = time.time() - start
            ProfilingRecorder.record(stage_name, duration)
            return result
        return wrapper
    return decorator

