import json
import logging
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import psutil
from locust import HttpUser, TaskSet, between, events, task

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Performance monitoring setup
class PerformanceMonitor:
    def __init__(self, sample_interval: int = 10):
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.sample_interval = sample_interval
        self.start_time = time.time()
        self.last_sample_time = self.start_time

    def take_sample(self):
        current_time = time.time()
        if current_time - self.last_sample_time >= self.sample_interval:
            process = psutil.Process()
            # Memory in MB
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            # CPU percentage
            cpu_percent = process.cpu_percent()
            self.cpu_samples.append(cpu_percent)

            self.last_sample_time = current_time

            # Check for memory leak
            self.check_memory_leak()

    def check_memory_leak(self):
        if len(self.memory_samples) > 5:  # Need at least 5 samples
            # Calculate memory growth rate (MB/min)
            time_elapsed = (time.time() - self.start_time) / 60  # Convert to minutes
            if time_elapsed > 0:
                memory_growth = (
                    self.memory_samples[-1] - self.memory_samples[0]
                ) / time_elapsed
                if memory_growth > 5:  # Alert if growth > 5 MB/min
                    logger.warning(
                        f"Memory leak detected! Growth rate: {memory_growth:.2f} MB/min"
                    )


# Global performance monitor
perf_monitor = PerformanceMonitor()

# Response time tracking
response_times: Dict[str, List[float]] = {
    "/api/stats": [],
    "/api/ml-predict": [],
    "/ws": [],
}


def calculate_p95(response_times: List[float]) -> float:
    if not response_times:
        return 0
    return np.percentile(response_times, 95)


@events.request.add_listener
def request_handler(
    request_type, name, response_time, response_length, exception, **kwargs
):
    if name in response_times:
        response_times[name].append(response_time)

        # Check P95 latency threshold (only if we have enough samples)
        if len(response_times[name]) >= 10:
            p95 = calculate_p95(response_times[name])
            if p95 > 2000:  # 2s threshold in milliseconds
                logger.error(f"P95 latency threshold exceeded for {name}: {p95}ms")


@events.test_start.add_listener
def on_test_start(**kwargs):
    logger.info("Load test starting")


@events.test_stop.add_listener
def on_test_stop(**kwargs):
    logger.info("Load test complete")

    # Log final performance metrics
    for endpoint, times in response_times.items():
        p95 = calculate_p95(times)
        logger.info(f"Endpoint {endpoint} P95 latency: {p95}ms")

    if perf_monitor.memory_samples:
        avg_memory = sum(perf_monitor.memory_samples) / len(perf_monitor.memory_samples)
        logger.info(f"Average memory usage: {avg_memory:.2f} MB")

    if perf_monitor.cpu_samples:
        avg_cpu = sum(perf_monitor.cpu_samples) / len(perf_monitor.cpu_samples)
        logger.info(f"Average CPU usage: {avg_cpu:.2f}%")


class UserBehavior(TaskSet):
    def on_start(self):
        """Initialize user behavior"""
        pass

    @task(3)
    def get_stats(self):
        with self.client.get(
            "/api/stats", name="/api/stats", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Stats request failed: {response.status_code}")
            else:
                response.success()

        # Monitor system resources
        perf_monitor.take_sample()

    @task(2)
    def ml_predict(self):
        # Sample prediction request data
        data = {
            "race_id": "test_race_001",
            "dogs": [
                {"name": "Dog1", "stats": {"wins": 5, "races": 10}},
                {"name": "Dog2", "stats": {"wins": 3, "races": 8}},
            ],
        }

        with self.client.post(
            "/api/ml-predict", json=data, name="/api/ml-predict", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"ML prediction failed: {response.status_code}")
            else:
                response.success()

        perf_monitor.take_sample()

    @task(1)
    def websocket_simulation(self):
        # Simulate WebSocket-like behavior with HTTP endpoint
        start_time = time.time()

        with self.client.get(
            "/ws?action=subscribe&race_id=test_race_001",
            name="/ws",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"WebSocket simulation failed: {response.status_code}")
            else:
                response.success()
                end_time = time.time()

                # Record WebSocket-like response time manually
                ws_response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times["/ws"].append(ws_response_time)

                # Check for slow WebSocket queries
                if ws_response_time > 100:
                    logger.warning(f"Slow WebSocket query: {ws_response_time:.2f}ms")

        perf_monitor.take_sample()


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)  # 1-2 second wait between tasks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable query plan analysis in database
        with self.client.get("/api/enable-explain-analyze") as response:
            if response.status_code != 200:
                logger.warning("Failed to enable EXPLAIN ANALYZE sampling")
