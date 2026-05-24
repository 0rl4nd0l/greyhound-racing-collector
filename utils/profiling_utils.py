import json
import os
import threading
import time
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
        self.session_data = None
        self.session_start_time = None

    def start_session(self, race_id, model_version, n_dogs, method):
        self.session_data = {
            "race_id": race_id,
            "model_version": model_version,
            "n_dogs": n_dogs,
            "method": method,
            "start_time": datetime.now().isoformat(),
        }
        self.session_start_time = time.time()

    def end_session(self):
        if not self.session_start_time:
            return

        try:
            duration = time.time() - self.session_start_time
            if self.session_data is not None:
                self.session_data["duration"] = duration
                self.session_data["end_time"] = datetime.now().isoformat()

                with self._data_lock:
                    # Add session data to recordings for consistency
                    self.recordings.append(self.session_data)

                    # Create audit directory with UTC ISO timestamp
                    audit_ts = (
                        datetime.utcnow().isoformat().replace(":", "").replace(".", "")
                    )
                    audit_dir = f"audit_results/{audit_ts}"
                    os.makedirs(audit_dir, exist_ok=True)

                    # Separate stage recordings from session data
                    stage_recordings = [
                        rec for rec in self.recordings if "stage_name" in rec
                    ]
                    session_recordings = [
                        rec for rec in self.recordings if "race_id" in rec
                    ]

                    # Prepare collected data structure as specified
                    collected_data = {
                        "timestamps": [
                            rec.get("end_time", datetime.now().isoformat())
                            for rec in stage_recordings
                        ],
                        "stage_durations": {
                            rec["stage_name"]: rec["duration"]
                            for rec in stage_recordings
                        },
                        "meta": {
                            "session_info": self.session_data,
                            "total_stages": len(stage_recordings),
                            "session_duration": duration,
                            "session_start": self.session_data.get("start_time"),
                            "session_end": self.session_data.get("end_time"),
                        },
                    }

                    # Dump collected dict to perf_metrics.json
                    metrics_path = os.path.join(audit_dir, "perf_metrics.json")
                    with open(metrics_path, "w") as f:
                        json.dump(collected_data, f, indent=2)

                    # Append readable line per stage to audit.log
                    log_path = os.path.join(audit_dir, "audit.log")
                    with open(
                        log_path, "w"
                    ) as f:  # Use 'w' to create new file per session
                        f.write(f"=== Profiling Session Report ===\n")
                        f.write(
                            f"Session ID: {self.session_data.get('race_id', 'unknown')}\n"
                        )
                        f.write(
                            f"Model Version: {self.session_data.get('model_version', 'unknown')}\n"
                        )
                        f.write(
                            f"Method: {self.session_data.get('method', 'unknown')}\n"
                        )
                        f.write(
                            f"Dogs Count: {self.session_data.get('n_dogs', 'unknown')}\n"
                        )
                        f.write(f"Total Duration: {duration:.4f}s\n")
                        f.write(
                            f"Session Start: {self.session_data.get('start_time')}\n"
                        )
                        f.write(f"Session End: {self.session_data.get('end_time')}\n")
                        f.write(f"\n=== Stage Performance ===\n")

                        for rec in stage_recordings:
                            stage_name = rec.get("stage_name", "Unknown Stage")
                            stage_duration = rec.get("duration", 0)
                            f.write(
                                f"Stage: {stage_name:<30} Duration: {stage_duration:.4f}s\n"
                            )

                        if not stage_recordings:
                            f.write("No stage recordings found.\n")

            self.session_data = None
            self.session_start_time = None
        except Exception as e:
            # Graceful error handling to avoid affecting predictions
            print(
                f"Warning: ProfilingRecorder end_session error (gracefully handled): {e}"
            )

    def record(self, stage_name, duration):
        with self._data_lock:
            self.recordings.append(
                {
                    "stage_name": stage_name,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                }
            )

    def flush_to_file(self):
        with self._data_lock:
            if not self.recordings:
                return

            timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
            dir_name = f"audit_results/{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            file_path = os.path.join(dir_name, "perf_metrics.json")

            # Write JSON metrics file
            with open(file_path, "w") as f:
                json.dump(self.recordings, f, indent=2)

            # Also write to audit.log
            with open("audit.log", "a") as f:
                f.write(
                    f"[{datetime.now().isoformat()}] Profiling data written to {file_path}\n"
                )
                for record in self.recordings:
                    f.write(f"[{datetime.now().isoformat()}] {json.dumps(record)}\n")

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
