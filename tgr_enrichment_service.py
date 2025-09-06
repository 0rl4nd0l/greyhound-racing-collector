#!/usr/bin/env python3
"""
TGR Data Enrichment Service
===========================

Automated service for continuously enriching race data with TGR insights
and maintaining data quality through intelligent processing.
"""

import json
import logging
import queue
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnrichmentStatus(Enum):
    """Status enumeration for enrichment jobs."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class EnrichmentJob:
    """Data class for enrichment job specifications."""

    job_id: str
    dog_name: str
    priority: int
    job_type: str
    created_at: datetime
    status: EnrichmentStatus = EnrichmentStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    error_message: str = None
    estimated_duration: float = 5.0


class TGREnrichmentService:
    """Automated TGR data enrichment service."""

    def __init__(
        self,
        db_path: str = "greyhound_racing_data.db",
        max_workers: int = 2,
        batch_size: int = 10,
    ):
        self.db_path = db_path
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.job_queue = queue.PriorityQueue()
        self.workers = []
        self.running = False
        self.stats = {
            "jobs_processed": 0,
            "jobs_succeeded": 0,
            "jobs_failed": 0,
            "total_processing_time": 0,
            "service_start_time": None,
        }
        # Track job_ids already enqueued to avoid duplicates when polling DB
        self._queued_job_ids = set()
        # Retry/backoff defaults (can be overridden by DB settings)
        self.default_max_attempts = 3
        self.backoff_base_seconds = 30
        self.backoff_max_seconds = 600
        self.retry_jitter_seconds = 15

        # Initialize service components
        self._setup_service_tables()
        # Load settings from DB (may adjust max_workers/backoff)
        self._load_settings()
        self._load_pending_jobs()

    def _setup_service_tables(self):
        """Set up database tables for the enrichment service."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create enrichment jobs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tgr_enrichment_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    dog_name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    error_message TEXT,
                    estimated_duration REAL DEFAULT 5.0,
                    actual_duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """
            )

            # Create enrichment service log
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tgr_service_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_action TEXT NOT NULL,
                    job_id TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create performance metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tgr_service_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create enrichment settings table (key/value)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tgr_enrichment_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()

            logger.info("✅ Enrichment service tables initialized")

        except Exception as e:
            logger.error(f"Failed to setup service tables: {e}")
            raise

    def _load_pending_jobs(self):
        """Load pending jobs from database into queue."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT job_id, dog_name, job_type, priority, attempts, max_attempts,
                       error_message, estimated_duration, created_at
                FROM tgr_enrichment_jobs
                WHERE status = 'pending' AND attempts < max_attempts
                ORDER BY priority DESC, created_at ASC
            """
            )

            pending_jobs = cursor.fetchall()
            added = 0

            for job_data in pending_jobs:
                (
                    job_id,
                    dog_name,
                    job_type,
                    priority,
                    attempts,
                    max_attempts,
                    error_message,
                    est_duration,
                    created_at,
                ) = job_data

                if job_id in self._queued_job_ids:
                    continue

                job = EnrichmentJob(
                    job_id=job_id,
                    dog_name=dog_name,
                    priority=priority,
                    job_type=job_type,
                    created_at=datetime.fromisoformat(created_at),
                    attempts=attempts,
                    max_attempts=max(
                        max_attempts or self.default_max_attempts,
                        self.default_max_attempts,
                    ),
                    error_message=error_message,
                    estimated_duration=est_duration,
                )

                # Add to priority queue (negative priority for max-heap behavior)
                self.job_queue.put((-priority, time.time(), job))
                self._queued_job_ids.add(job_id)
                added += 1

            conn.close()

            logger.info(
                f"✅ Loaded {added} pending enrichment jobs (total in DB: {len(pending_jobs)})"
            )

        except Exception as e:
            logger.error(f"Failed to load pending jobs: {e}")

    def start_service(self):
        """Start the enrichment service with worker threads."""

        if self.running:
            logger.warning("Service is already running")
            return

        logger.info(
            f"🚀 Starting TGR enrichment service with {self.max_workers} workers"
        )

        self.running = True
        self.stats["service_start_time"] = datetime.now()

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread, name=f"TGRWorker-{i+1}", daemon=True
            )
            worker.start()
            self.workers.append(worker)

        # Log service start
        self._log_service_action(
            "service_started", details=f"Started with {self.max_workers} workers"
        )

        logger.info("✅ Enrichment service started successfully")

    def stop_service(self):
        """Stop the enrichment service gracefully."""

        if not self.running:
            logger.warning("Service is not running")
            return

        logger.info("🛑 Stopping TGR enrichment service...")

        self.running = False

        # Wait for workers to finish current jobs
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=30)

        # Log service stop
        self._log_service_action("service_stopped")

        logger.info("✅ Enrichment service stopped")

    def add_enrichment_job(
        self, dog_name: str, job_type: str = "comprehensive", priority: int = 5
    ) -> str:
        """Add a new enrichment job to the queue."""

        safe_dog = str(dog_name).replace(" ", "_")
        job_id = f"enrich_{safe_dog}_{int(time.time())}_{job_type}"

        try:
            # Add to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR IGNORE INTO tgr_enrichment_jobs
                (job_id, dog_name, job_type, priority, status, estimated_duration)
                VALUES (?, ?, ?, ?, 'pending', ?)
            """,
                [
                    job_id,
                    dog_name,
                    job_type,
                    priority,
                    self._estimate_job_duration(job_type),
                ],
            )

            if cursor.rowcount > 0:
                # Add to queue
                job = EnrichmentJob(
                    job_id=job_id,
                    dog_name=dog_name,
                    priority=priority,
                    job_type=job_type,
                    created_at=datetime.now(),
                    estimated_duration=self._estimate_job_duration(job_type),
                )

                self.job_queue.put((-priority, time.time(), job))
                self._queued_job_ids.add(job_id)

                conn.commit()
                logger.info(f"✅ Added enrichment job: {job_id}")
            else:
                logger.warning(
                    f"Job already exists for {dog_name} with type {job_type}"
                )

            conn.close()
            return job_id

        except Exception as e:
            logger.error(f"Failed to add enrichment job: {e}")
            return None

    def _estimate_job_duration(self, job_type: str) -> float:
        """Estimate job duration based on type."""

        duration_map = {
            "quick": 2.0,
            "standard": 5.0,
            "comprehensive": 10.0,
            "deep_analysis": 15.0,
        }

        return duration_map.get(job_type, 5.0)

    def _worker_thread(self):
        """Worker thread for processing enrichment jobs."""

        thread_name = threading.current_thread().name
        logger.info(f"🔧 {thread_name} started")

        while self.running:
            try:
                # Get job from queue with timeout
                try:
                    _, _, job = self.job_queue.get(timeout=5)
                except queue.Empty:
                    # Poll DB for new jobs when queue appears empty
                    try:
                        self._load_pending_jobs()
                    except Exception:
                        pass
                    continue

                # Process the job
                success = self._process_enrichment_job(job)

                # Update statistics
                self.stats["jobs_processed"] += 1
                if success:
                    self.stats["jobs_succeeded"] += 1
                else:
                    self.stats["jobs_failed"] += 1

                # Mark task as done
                self.job_queue.task_done()

            except Exception as e:
                logger.error(f"{thread_name} error: {e}")
                time.sleep(1)

        logger.info(f"🔧 {thread_name} stopped")

    def _process_enrichment_job(self, job: EnrichmentJob) -> bool:
        """Process a single enrichment job."""

        start_time = time.time()
        thread_name = threading.current_thread().name

        logger.info(f"🔄 {thread_name} processing job {job.job_id}")

        try:
            # Update job status to processing
            self._update_job_status(
                job.job_id, EnrichmentStatus.PROCESSING, started_at=datetime.now()
            )

            # Log job start
            self._log_service_action(
                "job_started",
                job.job_id,
                f"Processing {job.job_type} for {job.dog_name}",
            )

            # Process based on job type
            if job.job_type == "comprehensive":
                success = self._process_comprehensive_enrichment(job)
            elif job.job_type == "performance_analysis":
                success = self._process_performance_analysis(job)
            elif job.job_type == "expert_insights":
                success = self._process_expert_insights(job)
            else:
                success = self._process_standard_enrichment(job)

            duration = time.time() - start_time
            self.stats["total_processing_time"] += duration

            if success:
                # Mark job as completed
                self._update_job_status(
                    job.job_id,
                    EnrichmentStatus.COMPLETED,
                    completed_at=datetime.now(),
                    actual_duration=duration,
                )

                self._log_service_action(
                    "job_completed",
                    job.job_id,
                    f"Successfully processed in {duration:.1f}s",
                )

                logger.info(
                    f"✅ {thread_name} completed job {job.job_id} in {duration:.1f}s"
                )

                # Record performance metric
                self._record_metric("job_duration", duration, "seconds")
                # Remove from in-memory queued set
                try:
                    self._queued_job_ids.discard(job.job_id)
                except Exception:
                    pass

            else:
                # Handle job failure
                job.attempts += 1
                # Ensure max_attempts reflects default if job was created before settings
                job.max_attempts = max(
                    job.max_attempts or self.default_max_attempts,
                    self.default_max_attempts,
                )

                if job.attempts >= job.max_attempts:
                    # Mark as failed
                    self._update_job_status(
                        job.job_id,
                        EnrichmentStatus.FAILED,
                        error_message="Max attempts exceeded",
                    )

                    self._log_service_action(
                        "job_failed",
                        job.job_id,
                        f"Failed after {job.attempts} attempts",
                    )

                    logger.error(
                        f"❌ {thread_name} failed job {job.job_id} after {job.attempts} attempts"
                    )
                    # Remove from in-memory queued set
                    try:
                        self._queued_job_ids.discard(job.job_id)
                    except Exception:
                        pass
                else:
                    # Retry later with exponential backoff and jitter
                    self._update_job_status(
                        job.job_id, EnrichmentStatus.PENDING, attempts=job.attempts
                    )

                    base = max(1, int(self.backoff_base_seconds))
                    cap = max(base, int(self.backoff_max_seconds))
                    delay = min(base * (2 ** max(0, job.attempts - 1)), cap)
                    jitter = (
                        random.randint(0, int(self.retry_jitter_seconds))
                        if self.retry_jitter_seconds > 0
                        else 0
                    )
                    scheduled_time = time.time() + delay + jitter

                    # Re-add to queue with lower priority and scheduled time ordering
                    retry_job = job
                    retry_job.status = EnrichmentStatus.PENDING
                    # Lower the priority to de-prioritize failing job
                    new_priority = max(1, job.priority - 1)
                    self.job_queue.put((-(new_priority), scheduled_time, retry_job))

                    logger.warning(
                        f"⚠️ {thread_name} retrying job {job.job_id} (attempt {job.attempts + 1}) in ~{delay + jitter}s"
                    )

            return success

        except Exception as e:
            logger.error(f"❌ {thread_name} error processing job {job.job_id}: {e}")

            # Update job with error
            self._update_job_status(
                job.job_id, EnrichmentStatus.FAILED, error_message=str(e)
            )

            self._log_service_action("job_error", job.job_id, str(e))

            return False

    def _process_comprehensive_enrichment(self, job: EnrichmentJob) -> bool:
        """Process comprehensive enrichment for a dog."""

        try:
            # Simulate comprehensive enrichment processing
            dog_name = job.dog_name

            # 1. Update performance summary
            self._update_dog_performance_summary(dog_name)

            # 2. Process expert insights
            self._process_dog_expert_insights(dog_name)

            # 3. Update venue/distance analysis
            self._update_venue_distance_analysis(dog_name)

            # 4. Refresh feature cache
            self._refresh_dog_feature_cache(dog_name)

            time.sleep(2)  # Simulate processing time

            return True

        except Exception as e:
            logger.error(f"Comprehensive enrichment failed for {job.dog_name}: {e}")
            return False

    def _process_performance_analysis(self, job: EnrichmentJob) -> bool:
        """Process performance analysis for a dog."""

        try:
            # Update performance metrics
            self._update_dog_performance_summary(job.dog_name)
            time.sleep(1)
            return True

        except Exception as e:
            logger.error(f"Performance analysis failed for {job.dog_name}: {e}")
            return False

    def _process_expert_insights(self, job: EnrichmentJob) -> bool:
        """Process expert insights for a dog."""

        try:
            # Process expert commentary
            self._process_dog_expert_insights(job.dog_name)
            time.sleep(1)
            return True

        except Exception as e:
            logger.error(f"Expert insights processing failed for {job.dog_name}: {e}")
            return False

    def _process_standard_enrichment(self, job: EnrichmentJob) -> bool:
        """Process standard enrichment for a dog."""

        try:
            # Basic enrichment processing
            self._update_dog_performance_summary(job.dog_name)
            self._refresh_dog_feature_cache(job.dog_name)
            time.sleep(1)
            return True

        except Exception as e:
            logger.error(f"Standard enrichment failed for {job.dog_name}: {e}")
            return False

    def _update_dog_performance_summary(self, dog_name: str):
        """Update performance summary for a dog."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate performance data (simulate TGR analysis)
        import random

        performance_data = {
            "total_starts": random.randint(10, 50),
            "wins": random.randint(1, 10),
            "places": random.randint(3, 15),
            "consistency_score": random.uniform(60, 95),
            "form_trend": random.choice(["improving", "stable", "declining"]),
        }

        performance_data["win_percentage"] = (
            performance_data["wins"] / performance_data["total_starts"]
        ) * 100
        performance_data["place_percentage"] = (
            performance_data["places"] / performance_data["total_starts"]
        ) * 100

        cursor.execute(
            """
            INSERT OR REPLACE INTO tgr_dog_performance_summary
            (dog_name, performance_data, total_entries, wins, places, 
             win_percentage, place_percentage, consistency_score, form_trend, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                dog_name,
                json.dumps(performance_data),
                performance_data["total_starts"],
                performance_data["wins"],
                performance_data["places"],
                performance_data["win_percentage"],
                performance_data["place_percentage"],
                performance_data["consistency_score"],
                performance_data["form_trend"],
                datetime.now().isoformat(),
            ],
        )

        conn.commit()
        conn.close()

    def _process_dog_expert_insights(self, dog_name: str):
        """Process expert insights for a dog."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate sample expert insight
        import random

        insights = [
            "Strong recent form with improved early pace",
            "Consistent performer who handles all track conditions well",
            "Best suited to longer distances with strong finishing kick",
            "Improving youngster with excellent box manners",
        ]

        insight = random.choice(insights)
        sentiment = random.uniform(-0.5, 0.8)

        cursor.execute(
            """
            INSERT OR IGNORE INTO tgr_expert_insights
            (dog_name, comment_type, comment_text, source, sentiment_score)
            VALUES (?, 'enriched_analysis', ?, 'automated_enrichment', ?)
        """,
            [dog_name, insight, sentiment],
        )

        conn.commit()
        conn.close()

    def _update_venue_distance_analysis(self, dog_name: str):
        """Update venue and distance analysis for a dog."""

        # Simulate venue/distance analysis updates
        pass

    def _refresh_dog_feature_cache(self, dog_name: str):
        """Refresh TGR feature cache for a dog."""

        try:
            from tgr_prediction_integration import TGRPredictionIntegrator

            integrator = TGRPredictionIntegrator(db_path=self.db_path)

            # Generate fresh features
            features = integrator._get_tgr_historical_features(dog_name, datetime.now())

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO tgr_enhanced_feature_cache
                (dog_name, race_timestamp, tgr_features, cached_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    dog_name,
                    datetime.now().isoformat(),
                    json.dumps(features),
                    datetime.now().isoformat(),
                    (datetime.now() + timedelta(hours=24)).isoformat(),
                ],
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.debug(f"Feature cache refresh failed for {dog_name}: {e}")

    def _update_job_status(
        self,
        job_id: str,
        status: EnrichmentStatus,
        started_at: datetime = None,
        completed_at: datetime = None,
        actual_duration: float = None,
        attempts: int = None,
        error_message: str = None,
    ):
        """Update job status in database."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build update query dynamically
            update_fields = ["status = ?"]
            params = [status.value]

            if started_at:
                update_fields.append("started_at = ?")
                params.append(started_at.isoformat())

            if completed_at:
                update_fields.append("completed_at = ?")
                params.append(completed_at.isoformat())

            if actual_duration is not None:
                update_fields.append("actual_duration = ?")
                params.append(actual_duration)

            if attempts is not None:
                update_fields.append("attempts = ?")
                params.append(attempts)

            if error_message:
                update_fields.append("error_message = ?")
                params.append(error_message)

            params.append(job_id)

            query = f"UPDATE tgr_enrichment_jobs SET {', '.join(update_fields)} WHERE job_id = ?"
            cursor.execute(query, params)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    def _log_service_action(self, action: str, job_id: str = None, details: str = None):
        """Log service action to database."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tgr_service_log (service_action, job_id, details)
                VALUES (?, ?, ?)
            """,
                [action, job_id, details],
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.debug(f"Failed to log service action: {e}")

    def _record_metric(self, metric_name: str, value: float, unit: str = None):
        """Record performance metric."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tgr_service_metrics (metric_name, metric_value, metric_unit)
                VALUES (?, ?, ?)
            """,
                [metric_name, value, unit],
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.debug(f"Failed to record metric: {e}")

    def _load_settings(self):
        """Load enrichment settings from the database into memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM tgr_enrichment_settings")
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return
            kv = {k: v for k, v in rows}
            # Parse known keys if present
            if kv.get("default_max_attempts"):
                try:
                    self.default_max_attempts = max(1, int(kv["default_max_attempts"]))
                except Exception:
                    pass
            if kv.get("backoff_base_seconds"):
                try:
                    self.backoff_base_seconds = max(1, int(kv["backoff_base_seconds"]))
                except Exception:
                    pass
            if kv.get("backoff_max_seconds"):
                try:
                    self.backoff_max_seconds = max(
                        self.backoff_base_seconds, int(kv["backoff_max_seconds"])
                    )
                except Exception:
                    pass
            if kv.get("retry_jitter_seconds"):
                try:
                    self.retry_jitter_seconds = max(0, int(kv["retry_jitter_seconds"]))
                except Exception:
                    pass
            if kv.get("concurrency_limit"):
                try:
                    self.max_workers = max(1, int(kv["concurrency_limit"]))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to load enrichment settings: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and statistics."""

        status = {
            "service_running": self.running,
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "queue_size": self.job_queue.qsize(),
            "statistics": self.stats.copy(),
        }

        # Calculate service uptime
        if self.stats["service_start_time"]:
            uptime = datetime.now() - self.stats["service_start_time"]
            status["uptime_seconds"] = uptime.total_seconds()

        # Get job status breakdown
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT status, COUNT(*) 
                FROM tgr_enrichment_jobs 
                GROUP BY status
            """
            )

            status_breakdown = dict(cursor.fetchall())
            status["job_status_breakdown"] = status_breakdown

            # Get recent job performance
            cursor.execute(
                """
                SELECT AVG(actual_duration), COUNT(*) 
                FROM tgr_enrichment_jobs 
                WHERE status = 'completed' 
                AND completed_at >= datetime('now', '-1 hour')
            """
            )

            recent_perf = cursor.fetchone()
            if recent_perf[1] > 0:
                status["recent_avg_duration"] = recent_perf[0]
                status["recent_jobs_completed"] = recent_perf[1]

            conn.close()

        except Exception as e:
            logger.error(f"Failed to get service status: {e}")

        return status

    def schedule_batch_enrichment(self, priority_dogs: List[str] = None) -> int:
        """Schedule batch enrichment for multiple dogs."""

        if not priority_dogs:
            # Get dogs from recent races that need enrichment
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT DISTINCT d.dog_clean_name
                    FROM dog_race_data d
                    LEFT JOIN tgr_dog_performance_summary ps ON d.dog_clean_name = ps.dog_name
                    WHERE d.dog_clean_name IS NOT NULL 
                    AND d.dog_clean_name != ''
                    AND (ps.last_updated IS NULL 
                         OR ps.last_updated < datetime('now', '-7 days'))
                    ORDER BY RANDOM()
                    LIMIT ?
                """,
                    [self.batch_size],
                )

                priority_dogs = [row[0] for row in cursor.fetchall()]
                conn.close()

            except Exception as e:
                logger.error(f"Failed to get priority dogs: {e}")
                return 0

        # Schedule jobs
        jobs_added = 0
        for dog_name in priority_dogs:
            job_id = self.add_enrichment_job(dog_name, "comprehensive", priority=5)
            if job_id:
                jobs_added += 1

        logger.info(f"✅ Scheduled {jobs_added} batch enrichment jobs")
        return jobs_added


def main():
    """Main service execution."""

    print("🔧 TGR Enrichment Service")
    print("-" * 30)

    service = TGREnrichmentService(max_workers=1, batch_size=5)

    try:
        # Start the service
        service.start_service()

        # Schedule some test jobs
        test_dogs = ["SWIFT THUNDER", "BALLARAT STAR", "RACING LEGEND"]
        for dog in test_dogs:
            service.add_enrichment_job(dog, "comprehensive", priority=8)

        # Let service run for a short period
        print("🔄 Running enrichment service for 30 seconds...")
        time.sleep(30)

        # Show service status
        status = service.get_service_status()
        print(f"\n📊 Service Status:")
        print(f"   Jobs Processed: {status['statistics']['jobs_processed']}")
        print(
            f"   Success Rate: {status['statistics']['jobs_succeeded']}/{status['statistics']['jobs_processed']}"
        )
        print(f"   Queue Size: {status['queue_size']}")
        print(f"   Uptime: {status.get('uptime_seconds', 0):.1f} seconds")

        if "job_status_breakdown" in status:
            print(f"   Job Status: {status['job_status_breakdown']}")

    finally:
        # Stop the service
        service.stop_service()

    print("\n✅ Enrichment service demo completed")


if __name__ == "__main__":
    main()
