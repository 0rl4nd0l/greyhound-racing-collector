#!/usr/bin/env python3
"""
Background Tasks for Greyhound Racing Analysis
==============================================

Defines background tasks for processing race data, file ingestion,
and ML predictions using Celery or RQ.

This module implements the background processing capabilities needed
for true end-to-end integration testing.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from celery import Celery

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    from redis import Redis
    from rq import Queue

    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False

from logger import logger

# Initialize task queues
celery_app = None
rq_queue = None

if CELERY_AVAILABLE:
    celery_app = Celery("greyhound_tasks")
    celery_app.config_from_object(
        {
            "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            "result_backend": os.getenv(
                "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
            ),
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            "timezone": "UTC",
            "enable_utc": True,
            "task_routes": {
                "tasks.process_race_file": {"queue": "default"},
                "tasks.generate_predictions": {"queue": "ml"},
                "tasks.download_race_data": {"queue": "scraping"},
            },
        }
    )

if RQ_AVAILABLE:
    redis_conn = Redis.from_url(os.getenv("RQ_REDIS_URL", "redis://localhost:6379/1"))
    rq_queue = Queue("default", connection=redis_conn)


def create_task_decorator(use_celery: bool = True):
    """Create a task decorator that works with either Celery or RQ"""

    def task_decorator(func):
        if use_celery and CELERY_AVAILABLE and celery_app:
            return celery_app.task(func)
        else:
            # Return function as-is for RQ or fallback
            return func

    return task_decorator


# Configure task decorator based on environment
USE_CELERY = os.getenv("USE_CELERY", "true").lower() == "true"
task = create_task_decorator(USE_CELERY)


@task
def process_race_file(
    file_path: str, metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Background task to process a race file through the ingestion pipeline.

    Args:
        file_path: Path to the race file to process
        metadata: Optional metadata about the race

    Returns:
        Dict containing processing results
    """
    logger.info(f"🏁 Starting background processing of race file: {file_path}")

    try:
        start_time = time.time()

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Race file not found: {file_path}")

        # Import processing modules
        try:
            from batch_prediction_pipeline import BatchPredictionPipeline
            from csv_ingestion import process_csv_file
        except ImportError as e:
            logger.error(f"Failed to import processing modules: {e}")
            return {
                "success": False,
                "error": f"Missing required modules: {e}",
                "file_path": file_path,
            }

        # Step 1: Process the CSV file through ingestion pipeline
        logger.info(f"📊 Processing CSV data for {file_path}")
        ingestion_result = process_csv_file(file_path)

        if not ingestion_result.get("success", False):
            raise Exception(
                f"CSV ingestion failed: {ingestion_result.get('error', 'Unknown error')}"
            )

        # Step 2: Extract race metadata
        race_id = ingestion_result.get("race_id")
        if not race_id:
            # Try to extract from filename
            filename = os.path.basename(file_path)
            race_id = filename.replace(".csv", "").replace("_", "-")

        # Step 3: Generate ML predictions if we have a prediction pipeline
        prediction_result = None
        try:
            if BatchPredictionPipeline:
                logger.info(f"🧠 Generating ML predictions for race {race_id}")
                pipeline = BatchPredictionPipeline()
                prediction_result = pipeline.predict_single_race(file_path)
        except Exception as pred_error:
            logger.warning(f"Prediction generation failed: {pred_error}")
            prediction_result = {"success": False, "error": str(pred_error)}

        processing_time = time.time() - start_time

        result = {
            "success": True,
            "file_path": file_path,
            "race_id": race_id,
            "processing_time": round(processing_time, 2),
            "ingestion_result": ingestion_result,
            "prediction_result": prediction_result,
            "rows_processed": ingestion_result.get("rows_processed", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"✅ Successfully processed race file {file_path} in {processing_time:.2f}s"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Error processing race file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
            "timestamp": datetime.utcnow().isoformat(),
        }


@task
def download_race_data(race_url: str, save_path: str = None) -> Dict[str, Any]:
    """
    Background task to download new race data from external sources.

    Args:
        race_url: URL to download race data from
        save_path: Optional path to save the downloaded data

    Returns:
        Dict containing download results
    """
    logger.info(f"📥 Starting background download of race data: {race_url}")

    try:
        start_time = time.time()

        # Import scraping modules
        try:
            from form_guide_csv_scraper import FormGuideCsvScraper
        except ImportError as e:
            logger.error(f"Failed to import scraping modules: {e}")
            return {
                "success": False,
                "error": f"Missing required scraping modules: {e}",
                "race_url": race_url,
            }

        # Initialize scraper
        scraper = FormGuideCsvScraper()

        # Download race data
        download_result = scraper.download_race_csv(race_url, save_path)

        if not download_result.get("success", False):
            raise Exception(
                f"Download failed: {download_result.get('error', 'Unknown error')}"
            )

        downloaded_file = download_result.get("file_path")
        download_time = time.time() - start_time

        # Automatically trigger processing if file was downloaded successfully
        process_task_id = None
        if downloaded_file and os.path.exists(downloaded_file):
            logger.info(
                f"🔄 Triggering background processing for downloaded file: {downloaded_file}"
            )
            if USE_CELERY and CELERY_AVAILABLE:
                process_task = process_race_file.delay(downloaded_file)
                process_task_id = process_task.id
            elif RQ_AVAILABLE and rq_queue:
                process_job = rq_queue.enqueue(process_race_file, downloaded_file)
                process_task_id = process_job.id

        result = {
            "success": True,
            "race_url": race_url,
            "file_path": downloaded_file,
            "download_time": round(download_time, 2),
            "file_size": os.path.getsize(downloaded_file) if downloaded_file else 0,
            "process_task_id": process_task_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"✅ Successfully downloaded race data from {race_url} in {download_time:.2f}s"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Error downloading race data from {race_url}: {e}")
        return {
            "success": False,
            "error": str(e),
            "race_url": race_url,
            "timestamp": datetime.utcnow().isoformat(),
        }


@task
def generate_predictions(
    race_ids: List[str], prediction_types: List[str] = None
) -> Dict[str, Any]:
    """
    Background task to generate ML predictions for specified races.

    Args:
        race_ids: List of race IDs to generate predictions for
        prediction_types: Types of predictions to generate

    Returns:
        Dict containing prediction results
    """
    if prediction_types is None:
        prediction_types = ["win_probability", "place_probability"]

    logger.info(
        f"🧠 Starting background prediction generation for {len(race_ids)} races"
    )

    try:
        start_time = time.time()

        # Import prediction modules
        try:
            from prediction_pipeline_v3 import PredictionPipelineV3

            from ml_system_v3 import MLSystemV3
        except ImportError as e:
            logger.error(f"Failed to import ML modules: {e}")
            return {
                "success": False,
                "error": f"Missing required ML modules: {e}",
                "race_ids": race_ids,
            }

        # Initialize prediction pipeline
        pipeline = PredictionPipelineV3()

        results = {}
        successful_predictions = 0

        for race_id in race_ids:
            logger.info(f"🎯 Generating predictions for race: {race_id}")

            try:
                # Generate predictions for this race
                prediction_result = pipeline.predict_race(race_id, prediction_types)

                if prediction_result.get("success", False):
                    successful_predictions += 1
                    results[race_id] = prediction_result

                    # Store predictions in database/cache for later retrieval
                    store_prediction_result(race_id, prediction_result)
                else:
                    results[race_id] = {
                        "success": False,
                        "error": prediction_result.get(
                            "error", "Unknown prediction error"
                        ),
                    }

            except Exception as race_error:
                logger.error(f"Error predicting race {race_id}: {race_error}")
                results[race_id] = {"success": False, "error": str(race_error)}

        processing_time = time.time() - start_time

        result = {
            "success": True,
            "race_ids": race_ids,
            "prediction_types": prediction_types,
            "total_races": len(race_ids),
            "successful_predictions": successful_predictions,
            "failed_predictions": len(race_ids) - successful_predictions,
            "processing_time": round(processing_time, 2),
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"✅ Generated predictions for {successful_predictions}/{len(race_ids)} races in {processing_time:.2f}s"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}")
        return {
            "success": False,
            "error": str(e),
            "race_ids": race_ids,
            "timestamp": datetime.utcnow().isoformat(),
        }


@task
def update_race_notes(race_id: str, notes: str, user_id: str = None) -> Dict[str, Any]:
    """
    Background task to update race notes and persist changes.

    Args:
        race_id: ID of the race to update
        notes: New notes content
        user_id: Optional user ID making the update

    Returns:
        Dict containing update results
    """
    logger.info(f"📝 Updating race notes for race: {race_id}")

    try:
        import sqlite3

        # Get database path
        database_path = os.getenv("DATABASE_PATH", "greyhound_racing_data.db")

        # Update notes in database
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()

            # Update or insert race notes
            cursor.execute(
                """
                INSERT OR REPLACE INTO race_notes (race_id, notes, updated_by, updated_at)
                VALUES (?, ?, ?, ?)
            """,
                (race_id, notes, user_id, datetime.utcnow().isoformat()),
            )

            conn.commit()

            # Verify the update
            cursor.execute(
                "SELECT notes, updated_at FROM race_notes WHERE race_id = ?", (race_id,)
            )
            result_row = cursor.fetchone()

            if result_row:
                updated_notes, updated_at = result_row

                logger.info(f"✅ Successfully updated notes for race {race_id}")
                return {
                    "success": True,
                    "race_id": race_id,
                    "notes": updated_notes,
                    "updated_at": updated_at,
                    "updated_by": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                raise Exception("Failed to verify note update")

    except Exception as e:
        logger.error(f"❌ Error updating race notes for {race_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "race_id": race_id,
            "timestamp": datetime.utcnow().isoformat(),
        }


def store_prediction_result(race_id: str, prediction_result: Dict[str, Any]) -> None:
    """Store prediction result in database for later retrieval"""
    try:
        import sqlite3

        database_path = os.getenv("DATABASE_PATH", "greyhound_racing_data.db")

        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(race_id)
                )
            """
            )

            # Store prediction result as JSON
            cursor.execute(
                """
                INSERT OR REPLACE INTO ml_predictions (race_id, prediction_data, created_at)
                VALUES (?, ?, ?)
            """,
                (race_id, json.dumps(prediction_result), datetime.utcnow().isoformat()),
            )

            conn.commit()
            logger.info(f"💾 Stored prediction result for race {race_id}")

    except Exception as e:
        logger.error(f"Failed to store prediction result for {race_id}: {e}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a background task"""
    try:
        if USE_CELERY and CELERY_AVAILABLE and celery_app:
            result = celery_app.AsyncResult(task_id)
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "info": result.info,
                "traceback": result.traceback,
            }
        elif RQ_AVAILABLE and rq_queue:
            from rq.job import Job

            job = Job.fetch(task_id, connection=rq_queue.connection)
            return {
                "task_id": task_id,
                "status": job.get_status(),
                "result": job.result if job.is_finished else None,
                "info": job.meta,
                "error": str(job.exc_info) if job.is_failed else None,
            }
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return None


def enqueue_task(task_func, *args, **kwargs) -> Optional[str]:
    """Enqueue a task using the configured task queue"""
    try:
        if USE_CELERY and CELERY_AVAILABLE:
            result = task_func.delay(*args, **kwargs)
            return result.id
        elif RQ_AVAILABLE and rq_queue:
            job = rq_queue.enqueue(task_func, *args, **kwargs)
            return job.id
        else:
            # Fallback: run synchronously
            logger.warning("No task queue available, running task synchronously")
            result = task_func(*args, **kwargs)
            return f"sync_{int(time.time())}"
    except Exception as e:
        logger.error(f"Error enqueuing task: {e}")
        return None


# Export main task functions for use in other modules
__all__ = [
    "process_race_file",
    "download_race_data",
    "generate_predictions",
    "update_race_notes",
    "get_task_status",
    "enqueue_task",
    "celery_app",
]
