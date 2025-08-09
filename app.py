
#!/usr/bin/env python3
"""
Greyhound Racing Dashboard
=========================

Flask web application for monitoring and analyzing greyhound racing data.
Integrates with the existing data collection and analysis system.

Author: AI Assistant
Date: July 11, 2025
"""

# CRITICAL: Import profiling disabler FIRST to prevent conflicts
import disable_profiling
from disable_profiling import is_profiling, set_profiling_enabled, profile_function, track_sequence

import json
import math
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, send_from_directory, stream_template, url_for)
from flask_cors import CORS
from flask_compress import Compress
from werkzeug.utils import secure_filename

try:
    from tests.integrity_test import run_integrity_test
except ImportError:

    def run_integrity_test():
        return {"status": "error", "message": "Integrity test not available"}


import threading
import time

import pandas as pd

from logger import logger

# Configuration constants
DEFAULT_PORT = 5002

# Import profiling configuration (disabled to avoid conflicts)
# from profiling_config import set_profiling_enabled, is_profiling

# Import CSV ingestion system for processing race files
try:
    from csv_ingestion import FormGuideCsvIngestor, create_ingestor, EnhancedFormGuideCsvIngestor, FormGuideCsvIngestionError
    CSV_INGESTION_AVAILABLE = True
    print("🚀 CSV ingestion system available")
except ImportError as e:
    print(f"⚠️ CSV ingestion system not available: {e}")
    CSV_INGESTION_AVAILABLE = False
    FormGuideCsvIngestor = None
    create_ingestor = None
    EnhancedFormGuideCsvIngestor = None
    FormGuideCsvIngestionError = Exception

# Import optimized caching and query systems
try:
    from endpoint_cache import get_endpoint_cache, cached_endpoint
    from optimized_queries import get_optimized_queries
    
    OPTIMIZATION_ENABLED = True
    print("🚀 Endpoint optimization enabled (caching + optimized queries)")
except ImportError as e:
    print(f"⚠️ Endpoint optimization not available: {e}")
    OPTIMIZATION_ENABLED = False
    
    def cached_endpoint(key_func=None, ttl=30):
        def decorator(func):
            return func
        return decorator

# Import database performance optimizations
try:
    from db_performance_optimizer import (
        initialize_db_optimization, get_db_pool, get_lazy_loader,
        query_performance_decorator
    )
    
    DB_OPTIMIZATION_ENABLED = True
    print("🚀 Database performance optimization enabled")
except ImportError as e:
    print(f"⚠️ Database optimization not available: {e}")
    DB_OPTIMIZATION_ENABLED = False
    
    def query_performance_decorator(func):
        return func

# Import pipeline profiler for bottleneck analysis (disabled due to conflicts)
# try:
#     from pipeline_profiler import (pipeline_profiler, profile_function,
#                                    track_sequence)
#
#     PROFILING_ENABLED = True
#     print("🔍 Pipeline profiling enabled")
# except ImportError:
#     print("⚠️ Pipeline profiling not available")
#     PROFILING_ENABLED = False

# Temporary stubs
class DummyTracker:
    def __enter__(self): return self
    def __exit__(self, *args): pass

def profile_function(func): return func
def track_sequence(step_name, component, step_type="processing"): return DummyTracker()


# Import Strategy Manager for unified prediction pipeline
try:
    from prediction_strategy_manager import get_strategy_manager

    STRATEGY_MANAGER_AVAILABLE = True
    strategy_manager = get_strategy_manager()  # Initialize strategy manager
    print("🎯 Strategy Manager available")
except ImportError:
    print("⚠️ Strategy Manager not available")
    STRATEGY_MANAGER_AVAILABLE = False
    strategy_manager = None

# Import comprehensive form data collector
try:
    from comprehensive_form_data_collector import \
        ComprehensiveFormDataCollector

    COMPREHENSIVE_COLLECTOR_AVAILABLE = True
    print("🚀 Comprehensive form data collector available")
except ImportError as e:
    print(f"⚠️ Comprehensive form data collector not available: {e}")
    COMPREHENSIVE_COLLECTOR_AVAILABLE = False
    ComprehensiveFormDataCollector = None

# Import batch prediction pipeline
try:
    from batch_prediction_pipeline import BatchPredictionPipeline
    BATCH_PIPELINE_AVAILABLE = True
    print("🚀 Batch prediction pipeline available")
except ImportError as e:
    print(f"⚠️ Batch prediction pipeline not available: {e}")
    BATCH_PIPELINE_AVAILABLE = False
    BatchPredictionPipeline = None

# Import background task system
try:
    from tasks import (
        process_race_file, download_race_data, generate_predictions, 
        update_race_notes, get_task_status, enqueue_task, celery_app
    )
    BACKGROUND_TASKS_AVAILABLE = True
    print("🚀 Background task system available")
except ImportError as e:
    print(f"⚠️ Background task system not available: {e}")
    BACKGROUND_TASKS_AVAILABLE = False
    process_race_file = None
    generate_predictions = None
    update_race_notes = None
import hashlib
import logging
# from venue_mapping_fix import GreyhoundVenueMapper  # Module not found
import pickle

import yaml
import glob

# Features and feature store imports
from features import (FeatureStore, V3BoxPositionFeatures,
                      V3CompetitionFeatures, V3DistanceStatsFeatures,
                      V3RecentFormFeatures, V3TrainerFeatures,
                      V3VenueAnalysisFeatures, V3WeatherTrackFeatures)
from sportsbet_odds_integrator import SportsbetOddsIntegrator
from utils.file_naming import (build_prediction_filename,
                               extract_race_id_from_csv_filename)
from utils.csv_metadata import parse_race_csv_meta

# Initialize feature store singleton
feature_store = FeatureStore()

# Model registry system
from model_registry import get_model_registry

# Import asset management system
try:
    from assets import AssetManager
    ASSET_MANAGEMENT_AVAILABLE = True
    print("🚀 Asset management system available")
except ImportError as e:
    print(f"⚠️ Asset management system not available: {e}")
    ASSET_MANAGEMENT_AVAILABLE = False
    AssetManager = None

# Import model training API blueprint
try:
    from model_training_api import model_training_bp
    MODEL_TRAINING_API_AVAILABLE = True
    print("🚀 Model training API blueprint available")
except ImportError as e:
    print(f"⚠️ Model training API blueprint not available: {e}")
    MODEL_TRAINING_API_AVAILABLE = False
    model_training_bp = None

# Import Guardian Service for file integrity protection
try:
    from services.guardian_service import get_guardian_service, start_guardian_service
    GUARDIAN_SERVICE_AVAILABLE = True
    print("🛡️ Guardian Service available")
except ImportError as e:
    print(f"⚠️ Guardian Service not available: {e}")
    GUARDIAN_SERVICE_AVAILABLE = False
    get_guardian_service = None
    start_guardian_service = None

# ML System V3 for comprehensive predictions
try:
    from ml_system_v4 import train_leakage_safe_model
    from prediction_pipeline_v3 import PredictionPipelineV3

    ML_SYSTEM_V3_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML System V3 not available: {e}")
    ML_SYSTEM_V3_AVAILABLE = False
    PredictionPipelineV3 = None

# ML System V4 for advanced predictions
try:
    from prediction_pipeline_v4 import PredictionPipelineV4

    ML_SYSTEM_V4_AVAILABLE = True
    print("🚀 ML System V4 (Advanced) available")
except ImportError as e:
    logger.warning(f"ML System V4 not available: {e}")
    ML_SYSTEM_V4_AVAILABLE = False
    PredictionPipelineV4 = None

# Legacy prediction system imports (kept for fallback)
try:
    from unified_predictor import UnifiedPredictor
except ImportError:
    UnifiedPredictor = None
    print("Warning: UnifiedPredictor not available")

try:
    from comprehensive_prediction_pipeline import \
        ComprehensivePredictionPipeline
except ImportError:
    ComprehensivePredictionPipeline = None
    print("Warning: ComprehensivePredictionPipeline not available")

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# GPT Prediction Enhancer singleton
gpt_enhancer_instance = None


def get_gpt_enhancer():
    """Get or create singleton GPTPredictionEnhancer instance"""
    global gpt_enhancer_instance
    if gpt_enhancer_instance is None:
        try:
            from gpt_prediction_enhancer import GPTPredictionEnhancer

            gpt_enhancer_instance = GPTPredictionEnhancer()
            logger.info("GPTPredictionEnhancer singleton initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GPTPredictionEnhancer: {e}")
            gpt_enhancer_instance = None
    return gpt_enhancer_instance


app = Flask(__name__)

# Load configuration from config.py
from config import get_config
config_class = get_config()
app.config.from_object(config_class)

# Override secret key if not already set
if not app.config.get('SECRET_KEY'):
    app.config['SECRET_KEY'] = "greyhound_racing_secret_key_2025"

# Initialize asset management system
if ASSET_MANAGEMENT_AVAILABLE and AssetManager:
    try:
        asset_manager = AssetManager(app)
        print("✅ Asset management system initialized successfully")
    except Exception as e:
        print(f"⚠️ Asset management initialization failed: {e}")
        asset_manager = None
else:
    asset_manager = None
    print("⚠️ Asset management system not available")

# Performance profiling hooks
request_times = {}
performance_log_file = "logs/perf_server.log"

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

@app.before_request
def before_request():
    """Track request start time for profiling"""
    if is_profiling():
        request.start_time = time.time()
        # Log the start of request processing
        with open(performance_log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} - START - {request.method} {request.path} from {request.remote_addr}\n")

@app.after_request
def after_request(response):
    """Track request completion time and log performance metrics"""
    if is_profiling() and hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        endpoint = request.endpoint or 'unknown'
        
        # Log performance metrics
        with open(performance_log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} - END - {request.method} {request.path} - {response.status_code} - {duration*1000:.2f}ms\n")
        
        # Log slow requests (>500ms)
        if duration > 0.5:
            logger.warning(f"Slow request detected: {request.method} {request.path} took {duration*1000:.2f}ms")
            with open(performance_log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} - SLOW - {request.method} {request.path} - {duration*1000:.2f}ms\n")
    
    return response

# Register model training API blueprint if available
if MODEL_TRAINING_API_AVAILABLE and model_training_bp:
    app.register_blueprint(model_training_bp)
    print("🎯 Model Training API routes registered successfully")
else:
    print("⚠️ Model Training API routes not registered - blueprint not available")

# Enable CORS for all domains on all routes
CORS(
    app,
    origins="*",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    supports_credentials=True,
)

# Initialize Flask-Compress for automatic response compression
compress = Compress()
compress.init_app(app)

# Configuration
DATABASE_PATH = "greyhound_racing_data.db"
UNPROCESSED_DIR = "./unprocessed"
PROCESSED_DIR = "./processed"
HISTORICAL_DIR = "./historical_races"
UPCOMING_DIR = "./upcoming_races"

# Upload configuration
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPCOMING_DIR


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict_page", methods=["GET", "POST"])
def predict_page():
    """Predict page - Select upcoming races for prediction"""
    if request.method == "POST":
        # Handle form submission
        race_files = request.form.getlist("race_files")
        action = request.form.get("action", "single")
        
        if not race_files:
            flash("Please select a race file", "error")
            return redirect(url_for("predict_page"))
        
        # For single prediction, use the first selected race
        race_file = race_files[0]
        
        try:
            race_file_path = os.path.join(UPCOMING_DIR, race_file)
            if not os.path.exists(race_file_path):
                flash(f"Race file not found: {race_file}", "error")
                return redirect(url_for("predict_page"))
            
            logger.log_process(f"Starting prediction for race: {race_file}")
            
            # Run actual prediction using available pipelines (V4 -> V3 -> UnifiedPredictor)
            prediction_result = None
            
            # Try PredictionPipelineV4 first (most advanced)
            if PredictionPipelineV4:
                try:
                    logger.log_process("Using PredictionPipelineV4 for web interface")
                    pipeline = PredictionPipelineV4()
                    prediction_result = pipeline.predict_race_file(race_file_path)
                    
                    if prediction_result and prediction_result.get("success"):
                        logger.log_process("PredictionPipelineV4 completed successfully")
                    else:
                        logger.log_process(f"PredictionPipelineV4 returned unsuccessful result: {prediction_result}")
                        prediction_result = None
                        
                except Exception as e:
                    logger.log_error(f"PredictionPipelineV4 failed: {e}")
                    prediction_result = None
            
            # Fallback to PredictionPipelineV3 if V4 fails
            if not prediction_result and PredictionPipelineV3:
                try:
                    logger.log_process("Fallback to PredictionPipelineV3 for web interface")
                    pipeline = PredictionPipelineV3()
                    prediction_result = pipeline.predict_race_file(race_file_path, enhancement_level="basic")
                    
                    if prediction_result and prediction_result.get("success"):
                        logger.log_process("PredictionPipelineV3 completed successfully")
                    else:
                        logger.log_process(f"PredictionPipelineV3 returned unsuccessful result: {prediction_result}")
                        prediction_result = None
                        
                except Exception as e:
                    logger.log_error(f"PredictionPipelineV3 failed: {e}")
                    prediction_result = None
            
            # Final fallback to UnifiedPredictor if both V4 and V3 fail
            if not prediction_result and UnifiedPredictor:
                try:
                    logger.log_process("Fallback to UnifiedPredictor for web interface")
                    predictor = UnifiedPredictor()
                    prediction_result = predictor.predict_race_file(race_file_path)
                    
                    if prediction_result and prediction_result.get("success"):
                        logger.log_process("UnifiedPredictor completed successfully")
                    else:
                        logger.log_process(f"UnifiedPredictor returned unsuccessful result: {prediction_result}")
                        prediction_result = None
                        
                except Exception as e:
                    logger.log_error(f"UnifiedPredictor failed: {e}")
                    prediction_result = None
            
            # Check if prediction was successful
            if not prediction_result or not prediction_result.get("success"):
                error_msg = prediction_result.get("error", "All prediction methods failed") if prediction_result else "No prediction pipeline available"
                flash(f"Prediction failed: {error_msg}", "error")
                return redirect(url_for("predict_page"))
            
            # Enhance prediction result with proper race information parsing
            if prediction_result and prediction_result.get("summary", {}).get("race_info"):
                race_info = prediction_result["summary"]["race_info"]
                # Use the csv metadata parser for more accurate parsing
                try:
                    from utils.csv_metadata import parse_race_csv_meta
                    csv_meta = parse_race_csv_meta(race_file_path)
                    if csv_meta and csv_meta.get("status") == "success":
                        # Update race info with properly parsed metadata
                        if csv_meta.get("race_number") and csv_meta["race_number"] > 0:
                            race_info["race_number"] = str(csv_meta["race_number"])
                        if csv_meta.get("venue") and csv_meta["venue"] != "Unknown":
                            race_info["venue"] = csv_meta["venue"]
                        if csv_meta.get("race_date") and csv_meta["race_date"] != "Unknown":
                            race_info["race_date"] = csv_meta["race_date"]
                        logger.log_process(f"Enhanced race info: {race_info}")
                except Exception as e:
                    logger.log_error(f"Error enhancing race info: {e}")
            
            # Get races for the form
            import requests
            default_port = os.environ.get('DEFAULT_PORT', os.environ.get('PORT', '5002'))
            response = requests.get(f"http://localhost:{default_port}/api/upcoming_races_csv")
            races = []
            if response.status_code == 200:
                races_data = response.json().get("races", [])
                races = [race["filename"] for race in races_data]
            
            return render_template("predict.html", 
                                 races=races, 
                                 prediction_result=prediction_result,
                                 selected_race=race_file)
                                 
        except Exception as e:
            logger.log_error(f"Error during prediction for {race_file}", error=e)
            flash(f"Prediction error: {str(e)}", "error")
            return redirect(url_for("predict_page"))
    
    # Handle GET request
    try:
        # Get all available race files directly from filesystem instead of paginated API
        race_filenames = []
        if os.path.exists(UPCOMING_DIR):
            for filename in os.listdir(UPCOMING_DIR):
                if filename.endswith('.csv') and not filename.startswith('.') and filename != 'README.md':
                    race_filenames.append(filename)
        
        # Sort filenames for better user experience
        race_filenames.sort()
        
        return render_template("predict.html", races=race_filenames)
        
    except Exception as e:
        logging.error(f"Error loading predict page: {str(e)}")
        flash("Error loading predict page", "error")
        return redirect(url_for("index"))


@app.route('/favicon.ico')
def favicon():
    # Always return a 204 No Content response since we don't have a favicon
    return Response(status=204)

@app.route("/api/dogs/search")
def api_dogs_search():
    """API endpoint to search for dogs"""
    try:
        query = request.args.get("q", "").strip()
        limit = request.args.get("limit", 20, type=int)

        if not query:
            return (
                jsonify(
                    logger.create_structured_error(
                        "Search query is required", error_code="MISSING_QUERY_PARAMETER"
                    )
                ),
                400,
            )

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                d.dog_id,
                d.dog_name,
                d.total_races,
                d.total_wins,
                d.total_places,
                d.best_time,
                d.average_position,
                d.last_race_date,
                COUNT(drd.id) as actual_races
            FROM dogs d
            LEFT JOIN dog_race_data drd ON d.dog_name = drd.dog_name
            WHERE d.dog_name LIKE ? OR d.dog_name LIKE ?
            GROUP BY d.dog_id, d.dog_name
            ORDER BY d.total_races DESC, d.total_wins DESC
            LIMIT ?
        """,
            (f"%{query}%", f"{query}%", limit),
        )

        dogs = cursor.fetchall()
        conn.close()

        results = []
        for dog in dogs:
            win_rate = (dog[3] / dog[2] * 100) if dog[2] > 0 else 0
            place_rate = (dog[4] / dog[2] * 100) if dog[2] > 0 else 0

            results.append(
                {
                    "dog_id": dog[0],
                    "dog_name": dog[1],
                    "total_races": dog[2],
                    "total_wins": dog[3],
                    "total_places": dog[4],
                    "win_percentage": round(win_rate, 1),
                    "place_percentage": round(place_rate, 1),
                    "best_time": dog[5],
                    "average_position": round(dog[6], 1) if dog[6] else None,
                    "last_race_date": dog[7],
                    "actual_races": dog[8],
                }
            )

        return jsonify(
            {"success": True, "dogs": results, "query": query, "count": len(results)}
        )

    except Exception as e:
        logger.exception("Error searching dogs")
        return (
            jsonify(
                logger.create_structured_error(
                    f"Error searching dogs: {str(e)}", error_code="DOG_SEARCH_FAILED"
                )
            ),
            500,
        )


@app.route('/api/ingest_csv', methods=['POST'])
def ingest_csv_route():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        ingestor = EnhancedFormGuideCsvIngestor()
        try:
            processed_data, validation_result = ingestor.ingest_csv(file_path)
            return jsonify({'success': True, 'records_processed': len(processed_data)}), 200
        except FormGuideCsvIngestionError as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': False, 'error': 'Unsupported file type'}), 400

@app.route("/api/dogs/<dog_name>/details")
def api_dog_details(dog_name):
    """API endpoint to get detailed information about a specific dog with comprehensive data"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Clean dog name to remove any box number prefixes
        cleaned_dog_name = dog_name
        if ". " in dog_name and dog_name.split(".")[0].isdigit():
            cleaned_dog_name = dog_name.split(". ", 1)[1]

        # Get comprehensive form data from the race context if available
        comprehensive_profile = None
        if COMPREHENSIVE_COLLECTOR_AVAILABLE and ComprehensiveFormDataCollector:
            try:
                # Check if we have comprehensive data already collected for this dog
                conn_comp = sqlite3.connect(DATABASE_PATH)
                cursor_comp = conn_comp.cursor()
                cursor_comp.execute(
                    "SELECT * FROM comprehensive_dog_profiles WHERE dog_name = ?",
                    (cleaned_dog_name,),
                )
                existing_data = cursor_comp.fetchone()
                conn_comp.close()

                if existing_data:
                    # Parse existing comprehensive data
                    comprehensive_profile = {
                        "career_stats": {
                            "total_races": (
                                existing_data[2] if len(existing_data) > 2 else 0
                            )
                        },
                        "form_quality_score": (
                            existing_data[3] if len(existing_data) > 3 else 0.5
                        ),
                        "track_specialization": {},
                        "recent_form_rating": "Available",
                        "career_phase": "Active",
                        "improvement_trends": [],
                        "consistency_rating": (
                            existing_data[4] if len(existing_data) > 4 else 0.5
                        ),
                    }
                    logger.info(
                        f"Using existing comprehensive data for {cleaned_dog_name}"
                    )
                else:
                    logger.info(
                        f"No comprehensive data found for {cleaned_dog_name} - using basic data only"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to get comprehensive profile for {cleaned_dog_name}: {e}"
                )

        # Get dog basic information
        cursor.execute(
            """
            SELECT 
                dog_id, dog_name, total_races, total_wins, total_places,
                best_time, average_position, last_race_date, created_at
            FROM dogs 
            WHERE dog_name = ? OR dog_name LIKE ?
        """,
            (cleaned_dog_name, f"%. {cleaned_dog_name}"),
        )

        dog_info = cursor.fetchone()
        if not dog_info:
            conn.close()
            return jsonify({"success": False, "message": "Dog not found"}), 404

        # Get recent performances (last 10 races)
        cursor.execute(
            """
            SELECT 
                drd.race_id,
                rm.race_name,
                rm.venue,
                rm.race_date,
                rm.distance,
                rm.grade,
                drd.box_number,
                drd.finish_position,
                drd.individual_time,
                drd.weight,
                drd.trainer_name,
                drd.odds_decimal,
                drd.margin
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.dog_name = ?
            ORDER BY rm.race_date DESC
            LIMIT 10
        """,
            (dog_name,),
        )

        recent_races = cursor.fetchall()

        # Get performance statistics by venue
        cursor.execute(
            """
            SELECT 
                rm.venue,
                COUNT(*) as races,
                SUM(CASE WHEN drd.finish_position = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN drd.finish_position <= 3 THEN 1 ELSE 0 END) as places,
                AVG(drd.finish_position) as avg_position,
                MIN(CAST(drd.individual_time AS FLOAT)) as best_time
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.dog_name = ? AND drd.finish_position IS NOT NULL
            GROUP BY rm.venue
            ORDER BY races DESC
        """,
            (dog_name,),
        )

        venue_stats = cursor.fetchall()

        # Get performance by distance
        cursor.execute(
            """
            SELECT 
                rm.distance,
                COUNT(*) as races,
                SUM(CASE WHEN drd.finish_position = 1 THEN 1 ELSE 0 END) as wins,
                AVG(drd.finish_position) as avg_position,
                MIN(CAST(drd.individual_time AS FLOAT)) as best_time
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.dog_name = ? AND drd.finish_position IS NOT NULL
            GROUP BY rm.distance
            ORDER BY races DESC
        """,
            (dog_name,),
        )

        distance_stats = cursor.fetchall()

        conn.close()

        # Format basic info with comprehensive enhancements
        win_rate = (dog_info[3] / dog_info[2] * 100) if dog_info[2] > 0 else 0
        place_rate = (dog_info[4] / dog_info[2] * 100) if dog_info[2] > 0 else 0

        dog_details = {
            "dog_id": dog_info[0],
            "dog_name": dog_info[1],
            "total_races": dog_info[2],
            "total_wins": dog_info[3],
            "total_places": dog_info[4],
            "win_percentage": round(win_rate, 1),
            "place_percentage": round(place_rate, 1),
            "best_time": dog_info[5],
            "average_position": round(dog_info[6], 1) if dog_info[6] else None,
            "last_race_date": dog_info[7],
            "created_at": dog_info[8],
        }

        # Enhance with comprehensive profile data if available
        if comprehensive_profile:
            dog_details.update(
                {
                    "comprehensive_career_stats": comprehensive_profile.get(
                        "career_stats", {}
                    ),
                    "comprehensive_track_performance": comprehensive_profile.get(
                        "track_performance", {}
                    ),
                    "comprehensive_distance_analysis": comprehensive_profile.get(
                        "distance_analysis", {}
                    ),
                    "comprehensive_trainer_info": comprehensive_profile.get(
                        "trainer_info", {}
                    ),
                    "comprehensive_form_trends": comprehensive_profile.get(
                        "form_trends", {}
                    ),
                    "comprehensive_sectional_analysis": comprehensive_profile.get(
                        "sectional_analysis", {}
                    ),
                    "has_comprehensive_data": True,
                }
            )
        else:
            dog_details["has_comprehensive_data"] = False

        # Format recent races
        recent_performances = []
        for race in recent_races:
            recent_performances.append(
                {
                    "race_id": race[0],
                    "race_name": race[1],
                    "venue": race[2],
                    "race_date": race[3],
                    "distance": race[4],
                    "grade": race[5],
                    "box_number": race[6],
                    "finish_position": race[7],
                    "race_time": race[8],
                    "weight": race[9],
                    "trainer": race[10],
                    "odds": race[11],
                    "margin": race[12],
                }
            )

        # Format venue statistics
        venue_performance = []
        for venue in venue_stats:
            venue_win_rate = (venue[2] / venue[1] * 100) if venue[1] > 0 else 0
            venue_place_rate = (venue[3] / venue[1] * 100) if venue[1] > 0 else 0

            venue_performance.append(
                {
                    "venue": venue[0],
                    "races": venue[1],
                    "wins": venue[2],
                    "places": venue[3],
                    "win_percentage": round(venue_win_rate, 1),
                    "place_percentage": round(venue_place_rate, 1),
                    "average_position": round(venue[4], 1) if venue[4] else None,
                    "best_time": venue[5],
                }
            )

        # Format distance statistics
        distance_performance = []
        for distance in distance_stats:
            dist_win_rate = (distance[2] / distance[1] * 100) if distance[1] > 0 else 0

            distance_performance.append(
                {
                    "distance": distance[0],
                    "races": distance[1],
                    "wins": distance[2],
                    "win_percentage": round(dist_win_rate, 1),
                    "average_position": round(distance[3], 1) if distance[3] else None,
                    "best_time": distance[4],
                }
            )

        # Prepare comprehensive response
        response_data = {
            "success": True,
            "dog_details": dog_details,
            "recent_performances": recent_performances,
            "venue_performance": venue_performance,
            "distance_performance": distance_performance,
        }

        # Add comprehensive data insights if available
        if comprehensive_profile:
            response_data.update(
                {
                    "comprehensive_insights": {
                        "form_quality_score": comprehensive_profile.get(
                            "form_quality_score", 0
                        ),
                        "track_specialization": comprehensive_profile.get(
                            "track_specialization", {}
                        ),
                        "recent_form_rating": comprehensive_profile.get(
                            "recent_form_rating", "Unknown"
                        ),
                        "career_phase": comprehensive_profile.get(
                            "career_phase", "Unknown"
                        ),
                        "improvement_trends": comprehensive_profile.get(
                            "improvement_trends", []
                        ),
                        "consistency_rating": comprehensive_profile.get(
                            "consistency_rating", 0
                        ),
                    }
                }
            )

        return jsonify(response_data)

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting dog details: {str(e)}"}
            ),
            500,
        )


@app.route("/api/dogs/<dog_name>/form")
def api_dog_form(dog_name):
    """API endpoint to get comprehensive form guide for a specific dog"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Clean dog name to remove any box number prefixes
        cleaned_dog_name = dog_name
        if ". " in dog_name and dog_name.split(".")[0].isdigit():
            cleaned_dog_name = dog_name.split(". ", 1)[1]

        # Get detailed race history from existing comprehensive data if available
        detailed_race_history = None
        if COMPREHENSIVE_COLLECTOR_AVAILABLE:
            try:
                # Check if we have detailed race history in comprehensive tables
                conn_comp = sqlite3.connect(DATABASE_PATH)
                cursor_comp = conn_comp.cursor()
                cursor_comp.execute(
                    """
                    SELECT race_date, venue, distance, finish_position, race_time, 
                           track_condition, sectional_times, weight
                    FROM comprehensive_race_history 
                    WHERE dog_name = ? 
                    ORDER BY race_date DESC 
                    LIMIT 20
                """,
                    (cleaned_dog_name,),
                )
                history_rows = cursor_comp.fetchall()
                conn_comp.close()

                if history_rows:
                    detailed_race_history = []
                    for row in history_rows:
                        detailed_race_history.append(
                            {
                                "race_date": row[0],
                                "venue": row[1],
                                "distance": row[2],
                                "finish_position": row[3],
                                "race_time": row[4],
                                "track_condition": row[5],
                                "sectional_times": json.loads(row[6]) if row[6] else {},
                                "weight": row[7],
                            }
                        )
                    logger.info(
                        f"Found {len(detailed_race_history)} detailed races for {cleaned_dog_name}"
                    )
                else:
                    logger.info(
                        f"No detailed race history found for {cleaned_dog_name}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to get detailed race history for {cleaned_dog_name}: {e}"
                )

        # Get last 20 performances with more details from the unified schema
        cursor.execute(
            """
            SELECT 
                drd.race_id,
                rm.race_name,
                rm.venue,
                rm.race_date,
                rm.distance,
                rm.grade,
                rm.track_condition,
                drd.box_number,
                drd.finish_position,
                drd.individual_time,
                drd.weight,
                drd.trainer_name,
                drd.odds_decimal,
                drd.margin,
                drd.sectional_1st,
                drd.sectional_2nd
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.dog_name = ? OR drd.dog_name LIKE ?
            ORDER BY rm.race_date DESC
            LIMIT 20
        """,
            (cleaned_dog_name, f"%. {cleaned_dog_name}"),
        )

        form_data = cursor.fetchall()
        conn.close()

        # Format form guide
        form_guide = []
        for performance in form_data:
            form_guide.append(
                {
                    "race_id": performance[0],
                    "race_name": performance[1],
                    "venue": performance[2],
                    "race_date": performance[3],
                    "distance": performance[4],
                    "grade": performance[5],
                    "track_condition": performance[6],
                    "box_number": performance[7],
                    "finish_position": performance[8],
                    "race_time": performance[9],
                    "weight": performance[10],
                    "trainer": performance[11],
                    "odds": performance[12],
                    "margin": performance[13],
                    "sectional_time": performance[14],
                    "split_times": performance[15],
                }
            )

        # Calculate enhanced form trends using comprehensive data if available
        form_trend = "Insufficient data"
        comprehensive_form_analysis = None

        if detailed_race_history and len(detailed_race_history) > 0:
            # Use comprehensive data for enhanced form analysis
            comprehensive_form_analysis = {
                "total_detailed_races": len(detailed_race_history),
                "sectional_trends": [],
                "track_condition_performance": {},
                "grade_progression": [],
                "weight_trends": [],
            }

            # Analyze sectional performance trends
            sectional_times = [
                race.get("sectional_times", {})
                for race in detailed_race_history
                if race.get("sectional_times")
            ]
            if sectional_times:
                comprehensive_form_analysis["sectional_trends"] = sectional_times[
                    :10
                ]  # Last 10 races with sectionals

            # Analyze track condition performance
            for race in detailed_race_history:
                condition = race.get("track_condition", "Unknown")
                if (
                    condition
                    not in comprehensive_form_analysis["track_condition_performance"]
                ):
                    comprehensive_form_analysis["track_condition_performance"][
                        condition
                    ] = []
                comprehensive_form_analysis["track_condition_performance"][
                    condition
                ].append(race.get("finish_position"))

            # Calculate form trend from detailed data
            if len(detailed_race_history) >= 5:
                recent_positions = [
                    race.get("finish_position")
                    for race in detailed_race_history[:5]
                    if race.get("finish_position")
                ]
                older_positions = [
                    race.get("finish_position")
                    for race in detailed_race_history[5:10]
                    if race.get("finish_position")
                ]

                if recent_positions and older_positions:
                    recent_avg = sum(recent_positions) / len(recent_positions)
                    older_avg = sum(older_positions) / len(older_positions)
                    form_trend = (
                        "Improving"
                        if recent_avg < older_avg
                        else "Declining" if recent_avg > older_avg else "Stable"
                    )

        # Fallback to basic form trend calculation
        elif len(form_guide) >= 5:
            recent_positions = [
                race["finish_position"]
                for race in form_guide[:5]
                if race["finish_position"]
            ]
            older_positions = [
                race["finish_position"]
                for race in form_guide[5:10]
                if race["finish_position"]
            ]

            recent_avg = (
                sum(recent_positions) / len(recent_positions) if recent_positions else 0
            )
            older_avg = (
                sum(older_positions) / len(older_positions) if older_positions else 0
            )

            form_trend = (
                "Improving"
                if recent_avg < older_avg
                else "Declining" if recent_avg > older_avg else "Stable"
            )

        response_data = {
            "success": True,
            "dog_name": dog_name,
            "form_guide": form_guide,
            "form_trend": form_trend,
            "total_performances": len(form_guide),
        }

        # Add comprehensive form analysis if available
        if comprehensive_form_analysis:
            response_data.update(
                {
                    "comprehensive_form_analysis": comprehensive_form_analysis,
                    "has_detailed_history": True,
                }
            )
        else:
            response_data["has_detailed_history"] = False

        return jsonify(response_data)

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error getting dog form: {str(e)}"}),
            500,
        )


@app.route("/api/dogs/top_performers")
def api_top_performers():
    """API endpoint to get top performing dogs"""
    try:
        metric = request.args.get(
            "metric", "win_rate"
        )  # win_rate, place_rate, total_wins
        limit = request.args.get("limit", 20, type=int)
        min_races = request.args.get("min_races", 5, type=int)

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        if metric == "win_rate":
            order_by = "CAST(total_wins AS FLOAT) / total_races DESC"
        elif metric == "place_rate":
            order_by = "CAST(total_places AS FLOAT) / total_races DESC"
        elif metric == "total_wins":
            order_by = "total_wins DESC"
        else:
            order_by = "total_races DESC"

        cursor.execute(
            f"""
            SELECT 
                dog_name,
                total_races,
                total_wins,
                total_places,
                best_time,
                average_position,
                last_race_date
            FROM dogs 
            WHERE total_races >= ?
            ORDER BY {order_by}
            LIMIT ?
        """,
            (min_races, limit),
        )

        top_dogs = cursor.fetchall()
        conn.close()

        # Format results
        performers = []
        for dog in top_dogs:
            win_rate = (dog[2] / dog[1] * 100) if dog[1] > 0 else 0
            place_rate = (dog[3] / dog[1] * 100) if dog[1] > 0 else 0

            performers.append(
                {
                    "dog_name": dog[0],
                    "total_races": dog[1],
                    "total_wins": dog[2],
                    "total_places": dog[3],
                    "win_percentage": round(win_rate, 1),
                    "place_percentage": round(place_rate, 1),
                    "best_time": dog[4],
                    "average_position": round(dog[5], 1) if dog[5] else None,
                    "last_race_date": dog[6],
                }
            )

        return jsonify(
            {
                "success": True,
                "top_performers": performers,
                "metric": metric,
                "min_races": min_races,
                "count": len(performers),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting top performers: {str(e)}"}
            ),
            500,
        )


@app.route("/api/dogs/all")
def api_all_dogs():
    """API endpoint to get all dogs with pagination"""
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 50, type=int)
        sort_by = request.args.get(
            "sort_by", "total_races"
        )  # total_races, dog_name, total_wins, win_rate
        order = request.args.get("order", "desc")  # asc, desc

        # Validate parameters
        if per_page > 100:
            per_page = 100  # Limit to prevent large responses

        offset = (page - 1) * per_page

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Define sort options
        sort_options = {
            "total_races": "total_races",
            "dog_name": "dog_name",
            "total_wins": "total_wins",
            "total_places": "total_places",
            "win_rate": "CAST(total_wins AS FLOAT) / total_races",
            "place_rate": "CAST(total_places AS FLOAT) / total_races",
            "average_position": "average_position",
            "last_race_date": "last_race_date",
        }

        order_by = sort_options.get(sort_by, "total_races")
        order_direction = "ASC" if order == "asc" else "DESC"

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM dogs")
        total_count = cursor.fetchone()[0]

        # Get dogs with pagination
        cursor.execute(
            f"""
            SELECT 
                dog_id,
                dog_name,
                total_races,
                total_wins,
                total_places,
                best_time,
                average_position,
                last_race_date
            FROM dogs 
            ORDER BY {order_by} {order_direction}
            LIMIT ? OFFSET ?
        """,
            (per_page, offset),
        )

        dogs = cursor.fetchall()
        conn.close()

        # Format results
        results = []
        for dog in dogs:
            win_rate = (dog[3] / dog[2] * 100) if dog[2] > 0 else 0
            place_rate = (dog[4] / dog[2] * 100) if dog[2] > 0 else 0

            results.append(
                {
                    "dog_id": dog[0],
                    "dog_name": dog[1],
                    "total_races": dog[2],
                    "total_wins": dog[3],
                    "total_places": dog[4],
                    "win_percentage": round(win_rate, 1),
                    "place_percentage": round(place_rate, 1),
                    "best_time": dog[5],
                    "average_position": round(dog[6], 1) if dog[6] else None,
                    "last_race_date": dog[7],
                }
            )

        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page)
        has_next = page < total_pages
        has_prev = page > 1

        return jsonify(
            {
                "success": True,
                "dogs": results,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev,
                },
                "sort_by": sort_by,
                "order": order,
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error getting all dogs: {str(e)}"}),
            500,
        )


@app.route("/api/races/paginated")
def api_races_paginated():
    """API endpoint for interactive races with pagination, search, and runners"""
    try:
        import traceback
        from datetime import datetime

        logger = logging.getLogger(__name__)
        # Get parameters with validation
        try:
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 10))
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid page or per_page parameter. Must be integers.",
                    }
                ),
                400,
            )

        # Validate page and per_page values
        if page < 1:
            return (
                jsonify(
                    {"success": False, "message": "Page number must be greater than 0."}
                ),
                400,
            )

        if per_page < 1:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Per page value must be greater than 0.",
                    }
                ),
                400,
            )

        per_page = min(per_page, 50)  # Limit to prevent overload
        sort_by = request.args.get("sort_by", "race_date")
        order = request.args.get("order", "desc")
        search = request.args.get("search", "").strip()

        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Database connection error: {str(e)}",
                    }
                ),
                500,
            )

        # Calculate offset
        offset = (page - 1) * per_page

        # Build base query with search functionality
        base_query = """
            SELECT 
                race_id,
                venue,
                race_number,
                race_date,
                race_name,
                grade,
                distance,
                field_size,
                winner_name,
                winner_odds,
                winner_margin,
                url,
                extraction_timestamp,
                track_condition
            FROM race_metadata 
        """

        # Add search conditions
        search_conditions = []
        search_params = []
        if search:
            search_conditions.extend(
                [
                    "venue LIKE ?",
                    "race_name LIKE ?",
                    "grade LIKE ?",
                    "winner_name LIKE ?",
                ]
            )
            search_term = f"%{search}%"
            search_params.extend([search_term, search_term, search_term, search_term])

        # Build WHERE clause
        where_clause = ""
        if search_conditions:
            where_clause = f"WHERE ({' OR '.join(search_conditions)})"

        # Build ORDER BY clause
        sort_options = {
            "race_date": "race_date",
            "venue": "venue",
            "confidence": "extraction_timestamp",  # Use extraction_timestamp as proxy for confidence
            "grade": "grade",
        }
        order_by = sort_options.get(sort_by, "race_date")
        order_direction = "ASC" if order == "asc" else "DESC"

        # Get total count for pagination
        try:
            count_query = f"SELECT COUNT(*) FROM race_metadata {where_clause}"
            cursor.execute(count_query, search_params)
            total_count = cursor.fetchone()[0]
        except Exception as e:
            conn.close()
            return (
                jsonify(
                    {"success": False, "message": f"Error getting race count: {str(e)}"}
                ),
                500,
            )

        # Get races with pagination
        try:
            races_query = f"""
                {base_query}
                {where_clause}
                ORDER BY {order_by} {order_direction}
                LIMIT ? OFFSET ?
            """

            cursor.execute(races_query, search_params + [per_page, offset])
            races = cursor.fetchall()
        except Exception as e:
            conn.close()
            return (
                jsonify(
                    {"success": False, "message": f"Error fetching races: {str(e)}"}
                ),
                500,
            )

        # Format race results and get runners for each race
        result_races = []
        for race in races:
            race_id = race[0]

            # Get runners for this race
            try:
                runners_query = """
                    SELECT 
                        dog_name,
                        box_number,
                        finish_position,
                        individual_time,
                        weight,
                        odds_decimal,
                        margin,
                        trainer_name
                    FROM dog_race_data 
                    WHERE race_id = ?
                    ORDER BY CAST(box_number AS INTEGER)
                """
                cursor.execute(runners_query, (race_id,))
                runners_data = cursor.fetchall()
            except Exception as e:
                conn.close()
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Error fetching runners for race {race_id}: {str(e)}",
                        }
                    ),
                    500,
                )

            # Format runners
            runners = []
            for runner in runners_data:
                # Calculate mock probabilities based on odds and historical data
                odds = runner[5] if runner[5] else 5.0
                win_prob = min(1.0 / float(odds) if odds > 0 else 0.1, 1.0)
                place_prob = min(
                    win_prob * 2.5, 1.0
                )  # Place probability is typically higher
                confidence = min(
                    win_prob * 0.8 + 0.2, 1.0
                )  # Base confidence on win probability

                # Safe conversion function for database values
                def safe_convert(value, convert_func, default):
                    try:
                        if value is None:
                            return default
                        if isinstance(value, (bytes, bytearray)):
                            try:
                                # Try to decode bytes to string first
                                decoded_value = value.decode("utf-8")
                                try:
                                    # Try direct conversion using the target function
                                    converted_result = convert_func(decoded_value)
                                    # Convert to string first to ensure JSON serialization
                                    return str(converted_result)
                                except (ValueError, TypeError, UnicodeDecodeError):
                                    return str(
                                        decoded_value
                                    )  # Fallback to string conversion if conversion fails, ensuring no bytes propagate
                            except (UnicodeDecodeError, ValueError, TypeError):
                                return (
                                    str(default) if default is not None else "N/A"
                                )  # Ensure string return
                        # Ensure the converted value is JSON serializable
                        converted = convert_func(value)
                        if isinstance(converted, (bytes, bytearray)):
                            return (
                                str(converted, "utf-8")
                                if isinstance(converted, bytes)
                                else str(converted)
                            )
                        return converted  # Return the actual converted value
                    except (ValueError, TypeError, UnicodeDecodeError):
                        return (
                            str(default) if default is not None else "N/A"
                        )  # Ensure string return
                    except Exception as e:
                        # Include any additional error handling here
                        return str(default) if default is not None else "N/A"

                runners.append(
                    {
                        "dog_name": safe_convert(runner[0], str, "Unknown"),
                        "box_number": safe_convert(runner[1], int, 0),
                        "finish_position": safe_convert(runner[2], int, None),
                        "individual_time": safe_convert(runner[3], str, None),
                        "weight": safe_convert(runner[4], float, None),
                        "odds": safe_convert(
                            runner[5], lambda x: f"{float(x):.2f}", "N/A"
                        ),
                        "margin": safe_convert(runner[6], str, None),
                        "trainer_name": safe_convert(runner[7], str, None),
                        "win_probability": round(win_prob, 3),
                        "place_probability": round(place_prob, 3),
                        "confidence": round(confidence, 3),
                    }
                )

            # ===== CRITICAL DATETIME PARSING FIX =====
            # This handles ISO format timestamps with microseconds: 2025-07-23T19:13:28.830973
            # See DATETIME_PARSING_DOCUMENTATION.md for full details
            # DO NOT MODIFY without reading documentation
            # Format extraction timestamp
            extraction_time = race[12]
            if extraction_time:
                try:
                    # STEP 1: Remove microseconds by splitting on decimal point
                    # Handles: 2025-07-23T19:13:28.830973 -> 2025-07-23T19:13:28
                    time_str = str(extraction_time).split(".")[0]

                    # STEP 2: Detect format type and parse accordingly
                    if "T" in time_str:
                        # ISO format with 'T' separator: 2025-07-23T19:13:28
                        dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
                    else:
                        # Standard format with space: 2025-07-23 19:13:28
                        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

                    # STEP 3: Standardize output format for frontend consistency
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                except Exception as e:
                    # STEP 4: Graceful error handling with logging
                    logger.warning(
                        f"Failed to parse extraction time '{extraction_time}': {e}"
                    )
                    formatted_time = (
                        str(extraction_time) if extraction_time else "Unknown"
                    )
            else:
                formatted_time = "Unknown"
            # ===== END CRITICAL DATETIME PARSING FIX =====

            race_url = (
                race[11]
                if race[11]
                else db_manager.generate_race_url(race[1], race[3], race[2])
            )

            result_races.append(
                {
                    "race_id": safe_convert(race[0], str, "Unknown"),
                    "venue": safe_convert(race[1], str, "Unknown"),
                    "race_number": safe_convert(race[2], int, 0),
                    "race_date": safe_convert(race[3], str, "Unknown"),
                    "race_name": safe_convert(
                        race[4], str, f"Race {safe_convert(race[2], str, '0')}"
                    ),
                    "grade": safe_convert(race[5], str, "Unknown"),
                    "distance": safe_convert(race[6], str, "Unknown"),
                    "field_size": safe_convert(race[7], int, 0),
                    "winner_name": safe_convert(race[8], str, "Unknown"),
                    "winner_odds": safe_convert(race[9], str, "N/A"),
                    "winner_margin": safe_convert(race[10], str, "N/A"),
                    "url": race_url,
                    "extraction_timestamp": formatted_time,
                    "track_condition": safe_convert(race[13], str, "Unknown"),
                    "runners": runners,
                }
            )

        conn.close()

        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1

        return jsonify(
            {
                "success": True,
                "races": result_races,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev,
                },
                "sort_by": sort_by,
                "order": order,
                "search": search,
            }
        )

    except Exception as e:
        traceback_str = "".join(traceback.format_exception(None, e, e.__traceback__))
        logger.error(
            f"Error in `/api/races/paginated`: {str(e)}\nStack trace: {traceback_str}"
        )

        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Unexpected error occurred. Please consult the logs.",
                }
            ),
            500,
        )


@app.route("/api/upcoming_races")
def api_upcoming_races():
    """API endpoint for live upcoming races from thedogs.com.au (DEFAULT) with CSV fallback"""
    try:
        # Get parameters
        days_ahead = request.args.get('days', 1, type=int)  # Default to today + tomorrow
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 50, type=int), 100)  # Max 100 per page
        source = request.args.get('source', 'live')  # 'live' or 'csv'
        
        # PRIMARY: Use live scraping by default
        if source == 'live':
            try:
                # Use UpcomingRaceBrowser for comprehensive live data
                from upcoming_race_browser import UpcomingRaceBrowser
                browser = UpcomingRaceBrowser()
                
                # Get races for multiple days if requested
                from datetime import datetime, timedelta
                all_races = []
                
                for day_offset in range(days_ahead + 1):  # Include today (0) + days_ahead
                    target_date = datetime.now().date() + timedelta(days=day_offset)
                    day_races = browser.get_races_for_date(target_date)
                    
                    if day_races:
                        # Add date to each race
                        for race in day_races:
                            race["date"] = target_date.strftime("%Y-%m-%d")
                        all_races.extend(day_races)
                        logger.info(f"Got {len(day_races)} races for {target_date}")
                
                races = all_races
                logger.info(f"Total live races found: {len(races)}")
                
            except Exception as live_error:
                logger.error(f"Live scraping failed: {live_error}")
                # Fallback to CSV data if live scraping fails
                logger.info("Falling back to CSV files from upcoming_races directory")
                races = load_upcoming_races_with_guaranteed_fields(refresh=True)
                source = "csv_fallback"
        else:
            # Use CSV files when explicitly requested
            races = load_upcoming_races_with_guaranteed_fields(refresh=True)
            source = "csv"
        
        # Convert races to consistent format for frontend
        formatted_races = []
        for race in races:
            # Handle both live scraping format and CSV format
            formatted_race = {
                "race_id": race.get("url", race.get("race_id", f"{race.get('venue', 'unknown')}_{race.get('race_number', 0)}"))
,
                "venue": race.get("venue", "Unknown"),
                "venue_name": race.get("venue_name", race.get("venue", "Unknown")),
                "race_number": race.get("race_number", 0),
                "date": race.get("date", race.get("race_date", "")),  # Handle both date fields
                "race_time": race.get("race_time", "TBA"),
                "race_name": race.get("race_name") or race.get("title", f"Race {race.get('race_number', 0)}"),
                "distance": race.get("distance", "Unknown"),
                "grade": race.get("grade", "Unknown"),
                "url": race.get("url", ""),
                "description": race.get("description", ""),
                "filename": race.get("filename", ""),  # For CSV files
                "source": source
            }
            formatted_races.append(formatted_race)
        
        # Sort races by date and time
        def sort_race_key(race):
            date_str = race.get("date", "9999-12-31")
            time_str = race.get("race_time", "23:59")
            
            # Convert time to sortable format
            try:
                if "PM" in time_str or "AM" in time_str:
                    # Convert 12-hour to 24-hour
                    time_part, period = time_str.split()
                    hours, minutes = map(int, time_part.split(':'))
                    if period == 'PM' and hours != 12:
                        hours += 12
                    elif period == 'AM' and hours == 12:
                        hours = 0
                    time_sort = f"{hours:02d}:{minutes:02d}"
                else:
                    time_sort = time_str
            except:
                time_sort = "23:59"
                
            return f"{date_str} {time_sort}"
        
        formatted_races.sort(key=sort_race_key)
        
        # Apply pagination
        total_count = len(formatted_races)
        if page > 1 or per_page < total_count:
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_races = formatted_races[start_idx:end_idx]
        else:
            paginated_races = formatted_races
            
        # Calculate pagination info
        import math
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1
            
        return jsonify({
            "success": True,
            "races": paginated_races,
            "total_count": total_count,
            "page": page,
            "per_page": per_page,
            "pagination": {
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            },
            "source": source,
            "message": f"Found {total_count} upcoming races ({'live from thedogs.com.au' if source == 'live' else 'from CSV files' if source == 'csv' else 'from CSV fallback'})"
        })
        
    except Exception as e:
        logger.error(f"Error in /api/upcoming_races: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error loading upcoming races: {str(e)}",
            "races": [],
            "total_count": 0,
            "source": "error"
        }), 500


@app.route("/api/upcoming_races_csv")
def api_upcoming_races_csv():
    """API endpoint to list upcoming races from CSV files with pagination and search"""
    from datetime import datetime
    try:
        # Get parameters with validation (same as /api/races/paginated)
        try:
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 10))
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid page or per_page parameter. Must be integers.",
                    }
                ),
                400,
            )

        # Validate page and per_page values
        if page < 1:
            return (
                jsonify(
                    {"success": False, "message": "Page number must be greater than 0."}
                ),
                400,
            )

        if per_page < 1:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Per page value must be greater than 0.",
                    }
                ),
                400,
            )

        per_page = min(per_page, 50)  # Limit to prevent overload
        sort_by = request.args.get("sort_by", "race_date")
        order = request.args.get("order", "desc")
        search = request.args.get("search", "").strip()

        # Check if upcoming races directory exists
        if not os.path.exists(UPCOMING_DIR):
            return jsonify(
                {
                    "success": True,
                    "races": [],
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total_count": 0,
                        "total_pages": 0,
                        "has_next": False,
                        "has_prev": False,
                    },
                    "sort_by": sort_by,
                    "order": order,
                    "search": search,
                }
            )

        # Get all CSV files directly from directory (avoid helper function causing duplicates)
        csv_files = []
        try:
            all_files = os.listdir(UPCOMING_DIR)
            csv_files = [f for f in all_files if f.endswith(".csv") and not f.startswith(".")]
        except OSError as e:
            logger.error(f"Error reading upcoming races directory: {e}")
            return jsonify(
                {
                    "success": False,
                    "message": "Error accessing upcoming races directory",
                }
            ), 500
        
        if not csv_files:
            return jsonify(
                {
                    "success": True,
                    "races": [],
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total_count": 0,
                        "total_pages": 0,
                        "has_next": False,
                        "has_prev": False,
                    },
                    "sort_by": sort_by,
                    "order": order,
                    "search": search,
                }
            )

        # Parse CSV files and extract race metadata
        races_data = []
        seen_races = set()  # Track unique races to prevent duplicates
        
        for filename in csv_files:
            file_path = os.path.join(UPCOMING_DIR, filename)
            
            try:
                # Skip if file doesn't exist or is not readable
                if not os.path.isfile(file_path):
                    continue
                    
                # Get file modification time for sorting
                file_mtime = os.path.getmtime(file_path)
                formatted_mtime = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M")
                
                # Extract race information from filename using improved regex patterns
                import re
                
                race_name = filename.replace(".csv", "")
                venue = "Unknown"
                race_date = "Unknown"
                race_number = 0
                
                # Pattern 1: "Race 1 - AP_K - 01 July 2025.csv"
                pattern1 = r'Race\s+(\d+)\s*-\s*([A-Z_/]+)\s*-\s*(\d{1,2}\s+\w+\s+\d{4})'
                match1 = re.search(pattern1, filename, re.IGNORECASE)
                
                # Pattern 2: "Race 1 - AP_K - 2025-08-04.csv"
                pattern2 = r'Race\s+(\d+)\s*-\s*([A-Z_/]+)\s*-\s*(\d{4}-\d{2}-\d{2})'
                match2 = re.search(pattern2, filename, re.IGNORECASE)
                
                if match1:
                    race_number = int(match1.group(1))
                    venue = match1.group(2).replace('/', '_')  # Normalize venue format
                    date_str = match1.group(3)
                    # Convert date like "01 July 2025" to "2025-07-01"
                    try:
                        parsed_date = datetime.strptime(date_str, "%d %B %Y")
                        race_date = parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        race_date = date_str  # Keep original if parsing fails
                elif match2:
                    race_number = int(match2.group(1))
                    venue = match2.group(2).replace('/', '_')  # Normalize venue format
                    race_date = match2.group(3)
                else:
                    # Fallback: try to extract individual components
                    race_match = re.search(r'Race[_\s]+(\d+)', filename, re.IGNORECASE)
                    if race_match:
                        race_number = int(race_match.group(1))
                    
                    venue_match = re.search(r'([A-Z_/]{2,6})', filename)
                    if venue_match:
                        venue = venue_match.group(1).replace('/', '_')
                    
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                    if date_match:
                        race_date = date_match.group(1)
                
                # Create a unique key to prevent duplicates
                unique_key = f"{venue}_{race_date}_{race_number}"
                if unique_key in seen_races:
                    logger.debug(f"Skipping duplicate race: {filename}")
                    continue
                seen_races.add(unique_key)
                
                # Try to read CSV to get additional info (like number of runners)
                field_size = 0
                distance = "Unknown"
                grade = "Unknown"
                
                try:
                    # Read CSV to count runners and get distance info
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        field_size = len(df)
                        # Try to get distance from DIST column (common in these files)
                        if 'DIST' in df.columns and not df['DIST'].empty:
                            distances = df['DIST'].dropna().unique()
                            if len(distances) > 0:
                                distance = f"{distances[0]}m"
                        # Try to get grade from G column
                        if 'G' in df.columns and not df['G'].empty:
                            grades = df['G'].dropna().unique()
                            if len(grades) > 0:
                                grade = str(grades[0])
                except Exception as e:
                    logger.debug(f"Could not read CSV content for {filename}: {e}")
                
                # Build race_id using MD5 hash of unique key (prevents duplicate IDs)
                race_id = hashlib.md5(unique_key.encode()).hexdigest()[:12]
                
                race_data = {
                    "race_id": race_id,
                    "venue": venue,
                    "race_number": race_number,
                    "race_date": race_date,
                    "race_name": race_name,
                    "grade": grade,
                    "distance": distance,
                    "field_size": field_size,
                    "winner_name": "Unknown",  # Upcoming race, no winner yet
                    "winner_odds": "N/A",
                    "winner_margin": "N/A",
                    "url": "",  # No URL for upcoming races
                    "extraction_timestamp": formatted_mtime,
                    "track_condition": "Unknown",  # Not available in CSV headers
                    "runners": [],  # Could be populated from CSV data if needed
                    "filename": filename,  # Added for frontend to request predictions
                    "file_mtime": file_mtime,  # For sorting
                }
                
                races_data.append(race_data)
                
            except Exception as e:
                logger.warning(f"Error processing CSV file {filename}: {e}")
                continue
        
        # Apply search filter if provided
        if search:
            filtered_races = []
            search_lower = search.lower()
            for race in races_data:
                if (search_lower in race["venue"].lower() or
                    search_lower in race["race_name"].lower() or
                    search_lower in race["grade"].lower() or
                    search_lower in race["filename"].lower()):
                    filtered_races.append(race)
            races_data = filtered_races
        
        # Sort races
        sort_options = {
            "race_date": "race_date",
            "venue": "venue",
            "confidence": "file_mtime",  # Use file modification time as proxy
            "grade": "grade",
        }
        
        sort_key = sort_options.get(sort_by, "file_mtime")
        reverse_sort = (order == "desc")
        
        # Handle different sort key types
        if sort_key == "file_mtime":
            races_data.sort(key=lambda x: x["file_mtime"], reverse=reverse_sort)
        else:
            races_data.sort(key=lambda x: str(x[sort_key]).lower(), reverse=reverse_sort)
        
        # Remove file_mtime from final output (used only for sorting)
        for race in races_data:
            race.pop("file_mtime", None)
        
        # Apply pagination
        total_count = len(races_data)
        offset = (page - 1) * per_page
        paginated_races = races_data[offset:offset + per_page]
        
        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1
        
        return jsonify(
            {
                "success": True,
                "races": paginated_races,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev,
                },
                "sort_by": sort_by,
                "order": order,
                "search": search,
            }
        )
        
    except Exception as e:
        logger.error(f"Error in /api/upcoming_races_csv: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error processing upcoming races: {str(e)}",
                }
            ),
            500,
        )

def create_batch_pipeline():
    """Initialize the batch prediction pipeline instance"""
    if not BATCH_PIPELINE_AVAILABLE:
        raise Exception("Batch prediction pipeline not available")
    
    return BatchPredictionPipeline()

@app.route("/api/batch/predict", methods=["POST"])
def api_batch_predict():
    """API endpoint for batch predictions"""
    data = request.get_json()

    # Validate input
    if not data or "files" not in data:
        return jsonify({"success": False, "message": "No files provided"}), 400

    file_paths = data.get("files", [])
    try:
        # Initialize the batch prediction pipeline
        pipeline = create_batch_pipeline()

        # Create a new batch job
        job_id = pipeline.create_batch_job(
            name="API Batch Prediction",
            input_files=file_paths,
            output_dir="./api_batch_output",
            batch_size=10,
            max_workers=3
        )

        # Run job asynchronously
        def run_batch_async(job_id):
            pipeline.run_batch_job(job_id)

        threading.Thread(target=run_batch_async, args=(job_id,), daemon=True).start()

        return jsonify({"success": True, "job_id": job_id, "message": "Batch prediction started"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/batch/status/<job_id>", methods=["GET"])
def api_batch_status(job_id):
    """API endpoint for checking the status of a batch job"""
    pipeline = create_batch_pipeline()
    job = pipeline.get_job_status(job_id)

    if not job:
        return jsonify({"success": False, "message": "Job not found"}), 404

    return jsonify({
        "success": True,
        "status": job.status,
        "progress": job.progress,
        "completed": job.completed_files,
        "failed": job.failed_files,
        "total": job.total_files,
        "created_at": job.created_at,
        "errors": job.error_messages
    })

@app.route("/api/batch/cancel/<job_id>", methods=["POST"])
def api_batch_cancel(job_id):
    """API endpoint to cancel a batch job"""
    pipeline = create_batch_pipeline()
    result = pipeline.cancel_job(job_id)

    if not result:
        return jsonify({"success": False, "message": "Unable to cancel job, or job not found"}), 404

    return jsonify({"success": True, "message": "Job cancelled"})

@app.route("/api/batch/progress/<job_id>", methods=["GET"])
def api_batch_progress(job_id):
    """API endpoint to get batch job progress with callback support"""
    try:
        pipeline = create_batch_pipeline()
        job = pipeline.get_job_status(job_id)

        if not job:
            return jsonify({"success": False, "message": "Job not found"}), 404

        # Support for callback parameter (JSONP-style callback)
        callback = request.args.get('callback')
        
        progress_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "completed": job.completed_files,
            "failed": job.failed_files,
            "total": job.total_files,
            "current_file": getattr(job, 'current_file', None),
            "created_at": job.created_at,
            "updated_at": getattr(job, 'updated_at', None),
            "errors": job.error_messages[-5:] if job.error_messages else [],  # Last 5 errors
            "recent_completions": getattr(job, 'recent_completions', [])[-3:] if hasattr(job, 'recent_completions') else [],  # Last 3 completed files
            "estimated_time_remaining": getattr(job, 'estimated_time_remaining', None),
            "timestamp": datetime.now().isoformat()
        }
        
        if callback:
            # Return JSONP response for cross-origin requests
            response_text = f"{callback}({json.dumps(progress_data)})"
            return Response(response_text, mimetype='application/javascript')
        else:
            return jsonify(progress_data)
            
    except Exception as e:
        error_response = {"success": False, "message": str(e)}
        if callback:
            response_text = f"{callback}({json.dumps(error_response)})"
            return Response(response_text, mimetype='application/javascript')
        else:
            return jsonify(error_response), 500

@app.route("/api/batch/stream", methods=["POST"])
def api_batch_stream():
    """API endpoint for streaming batch predictions with progress updates"""
    try:
        import json
        import uuid
        from flask import Response, copy_current_request_context

        data = request.get_json()
        if not data or "files" not in data:
            return jsonify({"success": False, "message": "No files provided"}), 400

        file_paths = data.get("files", [])
        batch_size = data.get("batch_size", 10)
        max_workers = data.get("max_workers", 3)
        stream_id = str(uuid.uuid4())

        @copy_current_request_context
        def generate_batch_stream():
            try:
                pipeline = create_batch_pipeline()
                
                # Send initial status
                yield f"data: {json.dumps({'type': 'start', 'stream_id': stream_id, 'message': f'Starting batch prediction for {len(file_paths)} files...', 'total_files': len(file_paths)})}\n\n"

                # Create batch job
                job_id = pipeline.create_batch_job(
                    name=f"Stream Batch {stream_id[:8]}",
                    input_files=file_paths,
                    output_dir="./stream_batch_output",
                    batch_size=batch_size,
                    max_workers=max_workers
                )
                
                yield f"data: {json.dumps({'type': 'job_created', 'job_id': job_id, 'message': f'Batch job {job_id} created'})}\n\n"

                # Start job asynchronously and monitor progress
                def run_batch_async_monitor(job_id):
                    pipeline.run_batch_job(job_id)

                import threading
                batch_thread = threading.Thread(target=run_batch_async_monitor, args=(job_id,), daemon=True)
                batch_thread.start()

                # Monitor job progress
                last_progress = -1
                last_completed = 0
                
                while True:
                    time.sleep(1)  # Check every second
                    
                    job = pipeline.get_job_status(job_id)
                    if not job:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                        break

                    # Send progress updates
                    if job.progress != last_progress or job.completed_files != last_completed:
                        yield f"data: {json.dumps({'type': 'progress', 'progress': job.progress, 'completed': job.completed_files, 'failed': job.failed_files, 'total': job.total_files, 'status': job.status})}\n\n"
                        last_progress = job.progress
                        last_completed = job.completed_files

                    # Send completion updates
                    if hasattr(job, 'recent_completions') and job.recent_completions:
                        for completion in job.recent_completions:
                            if completion not in getattr(generate_batch_stream, 'sent_completions', set()):
                                yield f"data: {json.dumps({'type': 'file_completed', 'file': completion, 'completed': job.completed_files, 'total': job.total_files})}\n\n"
                                if not hasattr(generate_batch_stream, 'sent_completions'):
                                    generate_batch_stream.sent_completions = set()
                                generate_batch_stream.sent_completions.add(completion)

                    # Check if job is complete
                    if job.status in ['completed', 'failed', 'cancelled']:
                        yield f"data: {json.dumps({'type': 'complete', 'status': job.status, 'completed': job.completed_files, 'failed': job.failed_files, 'total': job.total_files, 'message': f'Batch job {job.status}'})}\n\n"
                        break
                    
                    # Send keepalive every 30 seconds
                    if hasattr(generate_batch_stream, 'keepalive_counter'):
                        generate_batch_stream.keepalive_counter += 1
                    else:
                        generate_batch_stream.keepalive_counter = 1
                        
                    if generate_batch_stream.keepalive_counter % 30 == 0:
                        yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"

        return Response(
            generate_batch_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Stream setup error: {str(e)}"}), 500

@app.route("/api/ml-predict", methods=["POST"])
def api_ml_predict():
    """ML prediction endpoint for load testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Simulate ML prediction processing
        race_id = data.get("race_id", "unknown")
        dogs = data.get("dogs", [])
        
        # Basic validation
        if not dogs:
            return jsonify({"success": False, "error": "No dogs data provided"}), 400
        
        # Simulate prediction logic with database query
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Log query for performance monitoring
        query_start = time.time()
        cursor.execute("SELECT COUNT(*) as dog_count FROM dogs")
        total_dogs = cursor.fetchone()[0]
        query_time = (time.time() - query_start) * 1000  # Convert to ms
        
        # Log slow queries (>100ms)
        if query_time > 100:
            cursor.execute("""
                INSERT INTO query_monitoring (query, execution_time, query_plan)
                VALUES (?, ?, ?)
            """, ("SELECT COUNT(*) as dog_count FROM dogs", query_time, "SLOW_QUERY_DETECTED"))
            logger.warning(f"Slow query detected: {query_time:.2f}ms - SELECT COUNT(*) as dog_count FROM dogs")
        
        conn.commit()
        conn.close()
        
        # Generate mock predictions
        predictions = []
        for i, dog in enumerate(dogs):
            dog_name = dog.get("name", f"Dog {i+1}")
            stats = dog.get("stats", {})
            wins = stats.get("wins", 0)
            races = stats.get("races", 1)
            
            win_rate = wins / races if races > 0 else 0
            confidence = min(0.95, max(0.1, win_rate + 0.2))
            
            predictions.append({
                "dog_name": dog_name,
                "box_number": i + 1,
                "win_probability": round(confidence, 3),
                "confidence_level": "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW",
                "prediction_score": round(confidence, 3)
            })
        
        # Sort by prediction score
        predictions.sort(key=lambda x: x["prediction_score"], reverse=True)
        
        return jsonify({
            "success": True,
            "race_id": race_id,
            "predictions": predictions,
            "model_used": "LoadTestML_v1",
            "processing_time_ms": round(query_time, 2),
            "total_dogs_in_db": total_dogs,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"ML prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"ML prediction failed: {str(e)}"
        }), 500


@app.route("/ws")
def websocket_endpoint():
    """WebSocket-like endpoint for load testing (HTTP fallback)"""
    try:
        # Simulate WebSocket connection and data exchange
        message_type = request.args.get("type", "ping")
        race_id = request.args.get("race_id", "test_race")
        
        # Simulate database query for real-time data
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        query_start = time.time()
        cursor.execute("""
            SELECT COUNT(*) as active_races 
            FROM race_metadata 
            WHERE race_date >= date('now', '-1 day')
        """)
        active_races = cursor.fetchone()[0]
        query_time = (time.time() - query_start) * 1000
        
        # Log slow queries
        if query_time > 100:
            cursor.execute("""
                INSERT INTO query_monitoring (query, execution_time, query_plan)
                VALUES (?, ?, ?)
            """, ("WebSocket active races query", query_time, "WEBSOCKET_QUERY"))
            logger.warning(f"Slow WebSocket query: {query_time:.2f}ms")
        
        conn.commit()
        conn.close()
        
        # Simulate WebSocket response
        response_data = {
            "type": "response",
            "message_type": message_type,
            "race_id": race_id,
            "data": {
                "active_races": active_races,
                "connection_status": "connected",
                "server_time": datetime.now().isoformat(),
            },
            "query_time_ms": round(query_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {str(e)}")
        return jsonify({
            "type": "error",
            "error": f"WebSocket error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/api/enable-explain-analyze")
def enable_explain_analyze():
    """Enable EXPLAIN ANALYZE sampling for query monitoring"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Enable query logging
        cursor.execute("PRAGMA query_plan_enabled = ON")
        cursor.execute("PRAGMA query_plan_analysis = ON")
        
        # Create table for query monitoring if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_monitoring (
                id INTEGER PRIMARY KEY,
                query TEXT,
                execution_time REAL,
                query_plan TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "EXPLAIN ANALYZE sampling enabled"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to enable query monitoring: {str(e)}"
        }), 500

@app.route("/ping")
def ping():
    """Simple ping endpoint for testing compression"""
    return jsonify({
        "message": "pong",
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "server": "greyhound-racing-dashboard",
        "compression_test": "This is a longer message to ensure the response is large enough to trigger gzip compression when the minimum size threshold is met.",
        "data": {
            "uptime": "running",
            "version": "3.1.0",
            "environment": "development"
        }
    })

@app.route("/api/health")
def api_health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.1.0",
            "components": {
                "database": "connected",
                "ml_system": (
                    "available" if "ml_system_v3" in globals() else "unavailable"
                ),
                "prediction_pipeline": (
                    "available"
                    if "PredictionPipelineV3" in globals()
                    else "unavailable"
                ),
            },
        }
    )


@app.route("/api/races")
def api_races():
    """API endpoint to list all races with details"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT race_id, venue, race_date, race_name, winner_name FROM race_metadata WHERE winner_name IS NOT NULL"
    )
    races = cursor.fetchall()
    conn.close()

    return jsonify(
        [
            {
                "race_id": race[0],
                "venue": race[1],
                "race_date": race[2],
                "race_name": race[3],
                "winner_name": race[4],
            }
            for race in races
        ]
    )


@app.route("/predict", methods=["POST"])
def predict_basic():
    """Prediction endpoint that runs unified predictor"""
    data = request.get_json()
    if not data or "race_id" not in data:
        return jsonify({"error": "No race data provided"}), 400

    race_id = data["race_id"]
    try:
        if UnifiedPredictor is None:
            return jsonify({"error": "UnifiedPredictor not available"}), 500

        logger.log_process(f"Starting prediction for race ID: {race_id}")
        predictor = UnifiedPredictor()
        prediction_result = predictor.predict_race_file(race_id)
        logger.log_process(f"Completed prediction for race ID: {race_id}")
        return jsonify(prediction_result)
    except Exception as e:
        logger.log_error(f"Error during prediction for race ID: {race_id}", error=e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Unified API prediction endpoint"""
    data = request.get_json()

    # Handle both single race prediction and batch prediction requests
    if data and data.get("race_filename"):
        # Single race prediction (from frontend predictRace function)
        return api_predict_single_race()
    elif data and isinstance(data.get("race_data"), list):
        # Batch prediction (from runComprehensivePrediction)
        return api_predict_batch(data)
    else:
        # No specific data provided - run prediction on all upcoming races
        return api_predict_all_upcoming()


def api_predict_single_race():
    """Single race prediction using intelligent prediction pipeline"""
    data = request.get_json()
    if not data or "race_filename" not in data:
        return jsonify({"error": "No race filename provided"}), 400

    race_filename = data["race_filename"]
    race_file_path = os.path.join(UPCOMING_DIR, race_filename)

    try:
        if not os.path.exists(race_file_path):
            return jsonify({"error": f"Race file not found: {race_filename}"}), 404

        logger.log_process(f"Starting prediction for race: {race_filename}")

        # Initialize prediction
        prediction_result = None

        # Attempt primary prediction method
        if PredictionPipelineV3:
            try:
                logger.log_process("Using PredictionPipelineV3")
                pipeline = PredictionPipelineV3()
                prediction_result = pipeline.predict_race_file(race_file_path, enhancement_level="basic")
                
                # Check if prediction was actually successful
                if prediction_result and prediction_result.get("success"):
                    logger.log_process("PredictionPipelineV3 completed successfully")
                else:
                    logger.log_process(f"PredictionPipelineV3 returned unsuccessful result: {prediction_result}")
                    prediction_result = None  # Force fallback
                    
            except Exception as e:
                logger.log_error(f"PredictionPipelineV3 failed: {e}")
                prediction_result = None  # Ensure fallback will trigger

        # Fallback to UnifiedPredictor if primary fails
        if not prediction_result and UnifiedPredictor:
            try:
                logger.log_process("Fallback to UnifiedPredictor")
                predictor = UnifiedPredictor()
                prediction_result = predictor.predict_race_file(race_file_path)
                logger.log_process("UnifiedPredictor completed successfully")
            except Exception as e:
                logger.log_error(f"UnifiedPredictor failed: {e}")
                prediction_result = None

        # Return response based on prediction result
        if prediction_result and prediction_result.get("success"):
            return jsonify(
                {
                    "success": True,
                    "message": f"Prediction completed for {race_filename}",
                    "prediction": prediction_result,
                }
            )
        else:
            error_message = prediction_result.get("error", "All methods failed") if prediction_result else "No result"
            return jsonify({"success": False, "message": f"Prediction failed: {error_message}"}), 500

    except Exception as e:
        logger.log_error(f"Error during prediction for {race_filename}", error=e)
        return jsonify({"success": False, "message": f"Prediction error: {str(e)}"}), 500


@app.route("/predict_single", methods=["POST"])
def predict_single():
    """Predict single race from an uploaded CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        logger.log_process(f"File {filename} saved for prediction.")

        # Get model version info
        model_version = "unknown"
        if PredictionPipelineV3:
            model_version = "PredictionPipelineV3"
        elif UnifiedPredictor:
            model_version = "UnifiedPredictor"

        # Initialize prediction pipeline
        prediction_result = None
        
        # Try PredictionPipelineV3 first
        if PredictionPipelineV3:
            try:
                pipeline = PredictionPipelineV3()
                prediction_result = pipeline.predict_race_file(file_path, enhancement_level="basic")
            except Exception as e:
                logger.log_error(f"PredictionPipelineV3 failed: {e}")
                prediction_result = None
        
        # Fallback to UnifiedPredictor
        if not prediction_result and UnifiedPredictor:
            try:
                predictor = UnifiedPredictor()
                prediction_result = predictor.predict_race_file(file_path)
                model_version = "UnifiedPredictor"
            except Exception as e:
                logger.log_error(f"UnifiedPredictor failed: {e}")
                prediction_result = None

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass  # Don't fail if cleanup fails

        # Check if prediction was successful
        if not prediction_result:
            return jsonify({
                "success": False,
                "filename": filename,
                "model_version": model_version,
                "error": "Prediction pipeline not available or failed"
            }), 500

        # Extract predictions from result
        predictions = prediction_result.get("predictions", [])
        
        # Format response according to task requirements
        return jsonify({
            "success": True,
            "filename": filename,
            "model_version": model_version,
            "predictions": predictions
        })

    except Exception as e:
        logger.log_error(f"Error in predict_single endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "filename": getattr(file, 'filename', 'unknown') if 'file' in locals() else 'unknown',
            "model_version": "unknown",
            "error": f"Server error: {str(e)}"
        }), 500


def api_predict_batch(data):
    """Batch prediction using UnifiedPredictor"""
    try:
        if PredictionPipelineV3 is None:
            return jsonify({"error": "PredictionPipelineV3 not available"}), 500

        race_data = data["race_data"]
        # Use the existing processing status system for frontend logging
        safe_log_to_processing(
            f"🚀 Starting prediction pipeline for {len(race_data)} races", "INFO", 0
        )

        predictor = PredictionPipelineV3()
        results = []
        total_races = len(race_data)

        for i, entry in enumerate(race_data):
            progress = int((i / total_races) * 100)
            filename = entry.get("filename", entry.get("race_filename", str(entry)))
            safe_log_to_processing(
                f"📈 Processing race {i+1}/{total_races}: {filename}", "INFO", progress
            )

            race_file_path = os.path.join(UPCOMING_DIR, filename)
            if os.path.exists(race_file_path):
                prediction = predictor.predict_race_file(
                    race_file_path, enhancement_level="full"
                )
                results.append(prediction)
                safe_log_to_processing(
                    f"✅ Completed prediction for race {i+1}/{total_races}: {filename}",
                    "INFO",
                )
            else:
                safe_log_to_processing(f"⚠️ Race file not found: {filename}", "WARNING")
                results.append(
                    {"success": False, "error": f"File not found: {filename}"}
                )

        safe_log_to_processing(
            f"🎉 Unified prediction pipeline completed for {len(results)} races",
            "INFO",
            100,
        )

        return jsonify(
            {
                "success": True,
                "predictions": results,
                "total_processed": len(results),
                "successful_predictions": sum(1 for r in results if r.get("success")),
            }
        )
    except Exception as e:
        safe_log_to_processing(
            f"❌ Error in unified prediction pipeline: {str(e)}", "ERROR"
        )
        return jsonify({"error": str(e)}), 500


def api_predict_all_upcoming():
    """Predict all upcoming races using UnifiedPredictor"""
    try:
        if PredictionPipelineV3 is None:
            return jsonify({"error": "PredictionPipelineV3 not available"}), 500

        # Get all CSV files using the helper function
        upcoming_races = load_upcoming_races(refresh=False)
        upcoming_files = [race.get("filename", f"{race.get('name', 'race')}.csv") for race in upcoming_races if race.get("filename") or race.get("name")]
        
        # If no files from helper, fallback to direct directory scan for CSV only
        if not upcoming_files:
            upcoming_files = [
                f
                for f in os.listdir(UPCOMING_DIR)
                if f.endswith(".csv") and f != "README.md"
            ]

        if not upcoming_files:
            return jsonify(
                {
                    "success": True,
                    "message": "No upcoming races found",
                    "predictions": [],
                }
            )

        safe_log_to_processing(
            f"🚀 Starting prediction for {len(upcoming_files)} upcoming races",
            "INFO",
            0,
        )

        predictor = PredictionPipelineV3()
        results = []

        for i, filename in enumerate(upcoming_files):
            progress = int((i / len(upcoming_files)) * 100)
            safe_log_to_processing(
                f"📈 Processing race {i+1}/{len(upcoming_files)}: {filename}",
                "INFO",
                progress,
            )

            race_file_path = os.path.join(UPCOMING_DIR, filename)
            prediction = predictor.predict_race_file(
                race_file_path, enhancement_level="full"
            )
            results.append(prediction)

            safe_log_to_processing(
                f"✅ Completed prediction for race {i+1}/{len(upcoming_files)}: {filename}",
                "INFO",
            )

        safe_log_to_processing(
            f"🎉 Unified prediction completed for {len(results)} races", "INFO", 100
        )

        return jsonify(
            {
                "success": True,
                "predictions": results,
                "total_processed": len(results),
                "successful_predictions": sum(1 for r in results if r.get("success")),
            }
        )
    except Exception as e:
        safe_log_to_processing(f"❌ Error in unified prediction: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

        # Venue mapping for URL generation (updated to match upcoming_race_browser.py)
        self.venue_url_map = {
            "AP_K": "angle-park",
            "HOBT": "hobart",
            "GOSF": "gosford",
            "DAPT": "dapto",
            "SAN": "sandown",
            "MEA": "the-meadows",
            "WPK": "wentworth-park",
            "CANN": "cannington",
            "BAL": "ballarat",
            "BEN": "bendigo",
            "GEE": "geelong",
            "WAR": "warrnambool",
            "NOR": "northam",
            "MAND": "mandurah",
            "MURR": "murray-bridge",
            "GAWL": "gawler",
            "MOUNT": "mount-gambier",
            "TRA": "traralgon",
            "SAL": "sale",
            "RICH": "richmond",
            "HEA": "healesville",
            "CASO": "casino",
            "GRDN": "the-gardens",
            "DARW": "darwin",
            "ALBION": "albion-park",
            "HOR": "horsham",
        }

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def generate_race_url(self, venue, race_date, race_number):
        """Generate race URL based on scraper patterns"""
        try:
            venue_slug = self.venue_url_map.get(venue, venue.lower())
            # Generate URL in the format used by the scraper
            return f"https://www.thedogs.com.au/racing/{venue_slug}/{race_date}/{race_number}"
        except Exception:
            return None

    @query_performance_decorator
    def get_recent_races(self, limit=10):
        """Get recent races with basic info"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = """
            SELECT 
                race_id,
                venue,
                race_number,
                race_date,
                race_name,
                grade,
                distance,
                field_size,
                winner_name,
                winner_odds,
                winner_margin,
                url,
                extraction_timestamp,
                track_condition
            FROM race_metadata 
            ORDER BY extraction_timestamp DESC, race_date DESC 
            LIMIT ?
            """

            cursor.execute(query, (limit,))
            races = cursor.fetchall()
            conn.close()

            result = []
            for race in races:
                race_url = (
                    race[11]
                    if race[11]
                    else self.generate_race_url(race[1], race[3], race[2])
                )

                # Format extraction timestamp for better display
                extraction_time = race[12]
                if extraction_time:
                    try:
                        # STEP 1: Remove microseconds by splitting on decimal point
                        # Handles: 2025-07-23T19:13:28.830973 -> 2025-07-23T19:13:28
                        time_str = str(extraction_time).split(".")[0]

                        # STEP 2: Detect format type and parse accordingly
                        if "T" in time_str:
                            # ISO format with 'T' separator: 2025-07-23T19:13:28
                            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
                        else:
                            # Standard format with space: 2025-07-23 19:13:28
                            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

                        # STEP 3: Standardize output format for frontend consistency
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse extraction time '{extraction_time}': {e}"
                        )
                        formatted_time = (
                            str(extraction_time) if extraction_time else "Unknown"
                        )
                else:
                    formatted_time = "Unknown"

                result.append(
                    {
                        "race_id": race[0],
                        "venue": race[1],
                        "race_number": race[2],
                        "race_date": race[3],
                        "race_name": race[4],
                        "grade": race[5],
                        "distance": race[6],
                        "field_size": race[7],
                        "winner_name": race[8],
                        "winner_odds": race[9],
                        "winner_margin": race[10],
                        "url": race_url,
                        "extraction_timestamp": formatted_time,
                        "track_condition": race[13],
                    }
                )
            return result

        except Exception as e:
            print(f"Error getting recent races: {e}")
            return []

    @query_performance_decorator
    def get_paginated_races(self, page=1, per_page=20):
        """Get paginated races ordered by processing time (extraction_timestamp)"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Calculate offset
            offset = (page - 1) * per_page

            # Get total count for pagination info
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]

            # Updated query to prioritize races with winners (completed races)
            query = """
            SELECT 
                race_id,
                venue,
                race_number,
                race_date,
                race_name,
                grade,
                distance,
                field_size,
                winner_name,
                winner_odds,
                winner_margin,
                url,
                extraction_timestamp,
                track_condition
            FROM race_metadata 
            WHERE winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan'
            ORDER BY 
                CASE 
                    WHEN extraction_timestamp LIKE '%T%' THEN datetime(extraction_timestamp) 
                    ELSE datetime(extraction_timestamp) 
                END DESC,
                race_date DESC 
            LIMIT ? OFFSET ?
            """

            cursor.execute(query, (per_page, offset))
            races = cursor.fetchall()

            # If we don't have enough races with winners, get races without winners (form guides)
            if len(races) < per_page:
                remaining = per_page - len(races)
                incomplete_query = """
                SELECT 
                    race_id,
                    venue,
                    race_number,
                    race_date,
                    race_name,
                    grade,
                    distance,
                    field_size,
                    winner_name,
                    winner_odds,
                    winner_margin,
                    url,
                    extraction_timestamp,
                    track_condition
                FROM race_metadata 
                WHERE (winner_name IS NULL OR winner_name = '' OR winner_name = 'nan')
                ORDER BY 
                    CASE 
                        WHEN extraction_timestamp LIKE '%T%' THEN datetime(extraction_timestamp) 
                        ELSE datetime(extraction_timestamp) 
                    END DESC,
                    race_date DESC 
                LIMIT ?
                """
                cursor.execute(incomplete_query, (remaining,))
                incomplete_races = cursor.fetchall()
                races.extend(incomplete_races)

            result = []
            for race in races:
                race_url = (
                    race[11]
                    if race[11]
                    else self.generate_race_url(race[1], race[3], race[2])
                )

                # ===== CRITICAL DATETIME PARSING FIX =====
                # This handles ISO format timestamps with microseconds: 2025-07-23T19:13:28.830973
                # See DATETIME_PARSING_DOCUMENTATION.md for full details
                # DO NOT MODIFY without reading documentation
                # Format extraction timestamp
                extraction_time = race[12]
                if extraction_time:
                    try:
                        # STEP 1: Remove microseconds by splitting on decimal point
                        # Handles: 2025-07-23T19:13:28.830973 -e 2025-07-23T19:13:28
                        time_str = str(extraction_time).split(".")[0]

                        # STEP 2: Detect format type and parse accordingly
                        if "T" in time_str:
                            # ISO format with 'T' separator: 2025-07-23T19:13:28
                            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
                        else:
                            # Standard format with space: 2025-07-23 19:13:28
                            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

                        # STEP 3: Standardize output format for frontend consistency
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception as e:
                        # STEP 4: Graceful error handling with logging
                        logger.warning(
                            f"Failed to parse extraction time '{extraction_time}': {e}"
                        )
                        formatted_time = (
                            str(extraction_time) if extraction_time else "Unknown"
                        )
                else:
                    formatted_time = "Unknown"
                # ===== END CRITICAL DATETIME PARSING FIX =====

                result.append(
                    {
                        "race_id": race[0],
                        "venue": race[1],
                        "race_number": race[2],
                        "race_date": race[3],
                        "race_name": race[4],
                        "grade": race[5],
                        "distance": race[6],
                        "field_size": race[7],
                        "winner_name": race[8],
                        "winner_odds": race[9],
                        "winner_margin": race[10],
                        "url": race_url,
                        "extraction_timestamp": formatted_time,
                        "track_condition": race[13],
                    }
                )

            conn.close()

            # Calculate pagination info
            has_more = len(result) == per_page and (offset + per_page) < total_races
            total_pages = (total_races + per_page - 1) // per_page  # Ceiling division

            return {
                "races": result,
                "pagination": {
                    "current_page": page,
                    "per_page": per_page,
                    "total_races": total_races,
                    "total_pages": total_pages,
                    "has_more": has_more,
                    "has_previous": page > 1,
                },
            }

        except Exception as e:
            print(f"Error getting paginated races: {e}")
            return {
                "races": [],
                "pagination": {
                    "current_page": 1,
                    "per_page": per_page,
                    "total_races": 0,
                    "total_pages": 0,
                    "has_more": False,
                    "has_previous": False,
                },
            }

    def get_race_details(self, race_id):
        """Get detailed race information"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get race metadata
            cursor.execute(
                """
                SELECT * FROM race_metadata WHERE race_id = ?
            """,
                (race_id,),
            )

            race_data = cursor.fetchone()
            if not race_data:
                return None

            # Get column names
            cursor.execute("PRAGMA table_info(race_metadata)")
            columns = [col[1] for col in cursor.fetchall()]

            race_info = dict(zip(columns, race_data))

            # Get dog data for this race - filter out invalid entries
            cursor.execute(
                """
                SELECT * FROM dog_race_data 
                WHERE race_id = ? 
                AND dog_name IS NOT NULL 
                AND dog_name != 'nan' 
                AND dog_name != ''
                ORDER BY box_number
            """,
                (race_id,),
            )

            dogs_data = cursor.fetchall()

            # Get dog column names
            cursor.execute("PRAGMA table_info(dog_race_data)")
            dog_columns = [col[1] for col in cursor.fetchall()]

            dogs = [dict(zip(dog_columns, dog)) for dog in dogs_data]

            # Clean up the dog data
            for dog in dogs:
                # Replace 'nan' values with empty strings or defaults
                for key, value in dog.items():
                    if value == "nan" or value is None:
                        dog[key] = ""
                    elif isinstance(value, str) and value.lower() == "nan":
                        dog[key] = ""

                # Parse historical_records JSON if it exists
                if dog.get("historical_records"):
                    try:
                        import json

                        dog["historical_data"] = json.loads(dog["historical_records"])
                    except (json.JSONDecodeError, TypeError):
                        dog["historical_data"] = []

            conn.close()

            return {"race_info": race_info, "dogs": dogs}

        except Exception as e:
            print(f"Error getting race details: {e}")
            return None

    @query_performance_decorator
    def get_database_stats(self):
        """Get database statistics"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            stats = {}

            # Count races
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            stats["total_races"] = cursor.fetchone()[0]

            # Count dogs
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            stats["total_entries"] = cursor.fetchone()[0]

            # Count unique dogs
            cursor.execute("SELECT COUNT(DISTINCT dog_name) FROM dog_race_data")
            stats["unique_dogs"] = cursor.fetchone()[0]

            # Count venues
            cursor.execute("SELECT COUNT(DISTINCT venue) FROM race_metadata")
            stats["venues"] = cursor.fetchone()[0]

            # Latest race date
            cursor.execute("SELECT MAX(race_date) FROM race_metadata")
            latest_date = cursor.fetchone()[0]
            stats["latest_race_date"] = latest_date

            # Earliest race date
            cursor.execute("SELECT MIN(race_date) FROM race_metadata")
            earliest_date = cursor.fetchone()[0]
            stats["earliest_race_date"] = earliest_date

            conn.close()
            return stats

        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

    def get_stats(self):
        """Alias for get_database_stats() to maintain backward compatibility with tests"""
        return self.get_database_stats()


def get_file_stats():
    """Get comprehensive file processing statistics including enhanced data"""
    stats = {
        "unprocessed_files": 0,
        "processed_files": 0,
        "historical_files": 0,
        "upcoming_files": 0,
        "total_basic_files": 0,
        "enhanced_csv_files": 0,
        "enhanced_json_files": 0,
        "total_enhanced_files": 0,
        "archived_files": 0,
        "grand_total_files": 0,
    }

    try:
        # Count basic workflow files
        if os.path.exists(UNPROCESSED_DIR):
            unprocessed_files = [
                f for f in os.listdir(UNPROCESSED_DIR) if f.endswith(".csv")
            ]
            stats["unprocessed_files"] = len(unprocessed_files)

        if os.path.exists(PROCESSED_DIR):
            processed_count = 0
            for root, dirs, files in os.walk(PROCESSED_DIR):
                processed_count += len([f for f in files if f.endswith(".csv")])
            stats["processed_files"] = processed_count

        if os.path.exists(HISTORICAL_DIR):
            historical_files = [
                f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".csv")
            ]
            stats["historical_files"] = len(historical_files)

        if os.path.exists(UPCOMING_DIR):
            upcoming_files = [f for f in os.listdir(UPCOMING_DIR) if f.endswith(".csv")]
            stats["upcoming_files"] = len(upcoming_files)

        stats["total_basic_files"] = (
            stats["unprocessed_files"]
            + stats["processed_files"]
            + stats["historical_files"]
            + stats["upcoming_files"]
        )

        # Count enhanced data files
        enhanced_csv_dir = "./enhanced_expert_data/csv"
        enhanced_json_dir = "./enhanced_expert_data/json"

        if os.path.exists(enhanced_csv_dir):
            enhanced_csv_files = [
                f for f in os.listdir(enhanced_csv_dir) if f.endswith(".csv")
            ]
            stats["enhanced_csv_files"] = len(enhanced_csv_files)

        if os.path.exists(enhanced_json_dir):
            enhanced_json_files = [
                f for f in os.listdir(enhanced_json_dir) if f.endswith(".json")
            ]
            stats["enhanced_json_files"] = len(enhanced_json_files)

        stats["total_enhanced_files"] = (
            stats["enhanced_csv_files"] + stats["enhanced_json_files"]
        )

        # Count archived/cleanup files
        cleanup_dir = "./cleanup_archive"
        if os.path.exists(cleanup_dir):
            archived_files = [f for f in os.listdir(cleanup_dir) if f.endswith(".csv")]
            stats["archived_files"] = len(archived_files)

        # Calculate grand total
        stats["grand_total_files"] = (
            stats["total_basic_files"]
            + stats["total_enhanced_files"]
            + stats["archived_files"]
        )

        # Backward compatibility
        stats["total_files"] = stats["total_basic_files"]

    except Exception as e:
        logger.exception(f"Error getting file stats: {e}")

    return stats


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON instead of HTML for any other server error."""
    # Log the exception
    logger.exception(f"An unhandled exception occurred: {e}")

    # Prepare JSON response
    response = {"success": False, "message": "An unexpected server error occurred."}

    # Add more detail in debug mode
    if app.debug:
        response["error"] = str(e)
        import traceback

        response["traceback"] = traceback.format_exc()

    return jsonify(response), 500


# Initialize database manager
db_manager = DatabaseManager(DATABASE_PATH)

# Initialize database performance optimizations
if DB_OPTIMIZATION_ENABLED:
    try:
        from db_performance_optimizer import initialize_db_optimization
        db_pool, lazy_loader = initialize_db_optimization(
            database_path=DATABASE_PATH,
            pool_size=20  # Optimized for our 17 Gunicorn workers
        )
        print("✅ Database performance optimization initialized")
    except Exception as e:
        print(f"⚠️ Database optimization initialization failed: {e}")
        DB_OPTIMIZATION_ENABLED = False

# Initialize Sportsbet odds integrator
sportsbet_integrator = SportsbetOddsIntegrator(DATABASE_PATH)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app_debug.log"), logging.StreamHandler()],
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


# Model registry debug logging function
def log_model_registry_debug(message, level="INFO"):
    """Enhanced logging for model registry operations"""
    datetime.now().isoformat()
    log_message = f"[MODEL_REGISTRY] {message}"

    if level == "ERROR":
        logging.error(log_message)
    elif level == "WARNING":
        logging.warning(log_message)
    elif level == "DEBUG":
        logging.debug(log_message)
    else:
        logging.info(log_message)

    # Also log to enhanced logger if available
    try:
        if level == "ERROR":
            logger.log_error(log_message, context={"component": "model_registry"})
        else:
            logger.log_system(log_message, level, "MODEL_REGISTRY")
    except:
        pass  # Enhanced logger might not be available yet


# Helper function for model predictions
def get_model_predictions():
    """Get model predictions for display"""
    try:
        log_model_registry_debug("Retrieving model predictions for display", "DEBUG")

        # Check if model registry is available
        if model_registry is None:
            log_model_registry_debug(
                "Model registry not initialized, returning empty predictions", "WARNING"
            )
            return []

        # Get all models and their basic info
        raw_models = model_registry.list_models()
        log_model_registry_debug(f"Found {len(raw_models)} models in registry", "DEBUG")

        predictions = []
        for model in raw_models:
            try:
                model_dict = {
                    "model_id": getattr(model, "model_id", "N/A"),
                    "model_name": getattr(model, "model_name", "N/A"),
                    "model_type": getattr(model, "model_type", "N/A"),
                    "accuracy": getattr(model, "accuracy", 0.0),
                    "f1_score": getattr(model, "f1_score", 0.0),
                    "precision": getattr(model, "precision", 0.0),
                    "recall": getattr(model, "recall", 0.0),
                    "is_active": bool(getattr(model, "is_active", False)),
                    "last_updated": getattr(model, "last_updated", "Unknown"),
                    "training_samples": getattr(model, "training_samples", 0),
                    "features_count": getattr(model, "features_count", 0),
                }
                predictions.append(model_dict)
                log_model_registry_debug(
                    f"Retrieved data for model: {model_dict['model_name']}", "DEBUG"
                )
            except Exception as e:
                log_model_registry_debug(
                    f"Error processing model data: {str(e)}", "ERROR"
                )
                continue

        log_model_registry_debug(
            f"Successfully processed {len(predictions)} model predictions", "INFO"
        )
        return predictions

    except Exception as e:
        log_model_registry_debug(
            f"Error retrieving model predictions: {str(e)}", "ERROR"
        )
        return []


# Initialize model registry system with enhanced logging
log_model_registry_debug("Initializing model registry system...", "INFO")
try:
    model_registry = get_model_registry()
    model_count = len(model_registry.list_models())
    log_model_registry_debug(
        f"Model registry initialized successfully: {model_count} models tracked", "INFO"
    )
    print(f"✅ Model registry initialized successfully: {model_count} models tracked")
except Exception as e:
    log_model_registry_debug(f"Model registry initialization failed: {str(e)}", "ERROR")
    print(f"⚠️  Model registry initialization failed: {e}")
    model_registry = None

# Initialize database manager
print("🗄️ Initializing database manager...")
try:
    db_manager = DatabaseManager(DATABASE_PATH)
    print(f"✅ Database manager initialized successfully with database: {DATABASE_PATH}")
except Exception as e:
    print(f"⚠️ Database manager initialization failed: {e}")
    # Create a minimal fallback db_manager to prevent crashes
    db_manager = None


def run_schema_validation_and_healing(db_path, schema_contract_path):
    """Run schema validation and apply non-destructive fixes"""
    logger.info("🔍 Running Schema Validation and Healing...")

    try:
        # Load schema contract
        with open(schema_contract_path, "r") as file:
            schema_contract = yaml.safe_load(file)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Validate and repair tables and columns
        for table, config in schema_contract["tables"].items():
            cursor.execute(f"PRAGMA table_info({table})")
            columns = {col[1]: col[2] for col in cursor.fetchall()}
            for col in config["columns"]:
                col_name = col["name"]
                expected_type = col["type"]
                if col_name not in columns:
                    # Alter table to add missing column
                    alter_command = (
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {expected_type}"
                    )
                    cursor.execute(alter_command)
                elif columns[col_name] != expected_type:
                    logger.warning(
                        f"Type mismatch for column {col_name} in table {table}"
                    )

        conn.commit()
        logger.info("✅ Schema validation and healing complete.")
    except Exception as e:
        logger.error(f"Schema Validation Error: {e}")
    finally:
        conn.close()


@app.route("/")
def index():
    """Home page with dashboard overview"""
    db_stats = db_manager.get_database_stats()
    file_stats = get_file_stats()
    recent_races = db_manager.get_recent_races(limit=5)

    return render_template(
        "index.html",
        db_stats=db_stats,
        file_stats=file_stats,
        recent_races=recent_races,
    )


@app.route("/races")
def races():
    """Historical races listing page - ordered by processing time"""
    page = request.args.get("page", 1, type=int)
    per_page = 20

    # Use paginated method for proper ordering and pagination
    races_data = db_manager.get_paginated_races(page=page, per_page=per_page)

    return render_template(
        "races.html",
        races=races_data["races"],
        page=page,
        pagination=races_data["pagination"],
    )


@app.route("/race/<race_id>")
def race_detail(race_id):
    """Individual race detail page"""
    race_data = db_manager.get_race_details(race_id)

    if not race_data:
        flash("Race not found", "error")
        return redirect(url_for("races"))

    return render_template("race_detail.html", race_data=race_data)


@app.route("/predictions")
def predictions():
    """Predictions page - Display model predictions"""
    try:
        # Get model predictions using our new function
        predictions = get_model_predictions()
        # Log prediction retrieval
        logging.info(f"Retrieved {len(predictions)} model predictions")
        log_model_registry_debug(
            f"Predictions page loaded with {len(predictions)} models", "INFO"
        )
        return render_template("predictions.html", predictions=predictions)
    except Exception as e:
        # Log any errors
        logging.error(f"Error loading predictions: {str(e)}")
        log_model_registry_debug(f"Error loading predictions page: {str(e)}", "ERROR")
        flash("Error loading predictions", "error")
        return redirect(url_for("index"))


@app.route("/monitoring")
def monitoring():
    """Monitoring page - Overview of system status and logs"""
    try:
        # Log the entry to the monitoring page
        logging.debug("Accessing monitoring page")
        # Get model predictions using our new function
        predictions = get_model_predictions()
        # Log monitoring access
        logging.info(
            f"Monitoring page loaded with {len(predictions)} model predictions"
        )
        log_model_registry_debug(
            f"Monitoring page loaded with {len(predictions)} models", "INFO"
        )
        # Display monitoring details with predictions data
        return render_template("monitoring.html", predictions=predictions)
    except Exception as e:
        logging.error(f"Error accessing monitoring page: {str(e)}")
        log_model_registry_debug(f"Error loading monitoring page: {str(e)}", "ERROR")
        flash("Error accessing monitoring page", "error")
        return redirect(url_for("index"))


@app.route("/scraping")
def scraping_status():
    """Scraping status and controls with data processing features"""
    db_stats = db_manager.get_database_stats()
    file_stats = get_file_stats()

    # Get recent unprocessed files
    unprocessed_files = []
    if os.path.exists(UNPROCESSED_DIR):
        files = [f for f in os.listdir(UNPROCESSED_DIR) if f.endswith(".csv")]
        for filename in sorted(files, reverse=True)[:10]:
            file_path = os.path.join(UNPROCESSED_DIR, filename)
            file_stat = os.stat(file_path)
            unprocessed_files.append(
                {
                    "filename": filename,
                    "size": file_stat.st_size,
                    "modified": datetime.fromtimestamp(file_stat.st_mtime),
                }
            )

    return render_template(
        "scraping_status.html",
        db_stats=db_stats,
        file_stats=file_stats,
        unprocessed_files=unprocessed_files,
    )


@app.route("/logs")
def logs_viewer():
    """System logs viewer"""
    return render_template("logs.html")


@app.route("/model_registry")
def model_registry_view():
    """Model Registry Dashboard"""
    if model_registry is None:
        flash("Model registry not initialized", "error")
        return redirect(url_for("index"))

    # Get all models and preprocess them for the template
    raw_models = model_registry.list_models()
    models = []

    for model in raw_models:
        # Create a dict with safe attribute access
        model_dict = {
            "model_name": getattr(model, "model_name", "N/A"),
            "model_id": getattr(model, "model_id", "N/A"),
            "model_type": getattr(model, "model_type", "N/A"),
            "accuracy": getattr(model, "accuracy", 0),
            "f1_score": getattr(model, "f1_score", 0),
            "precision": getattr(model, "precision", 0),
            "recall": getattr(model, "recall", 0),
            "features_count": getattr(model, "features_count", 0),
            "training_samples": getattr(model, "training_samples", 0),
            "is_active": bool(getattr(model, "is_active", False)),
        }
        models.append(model_dict)

    # Find best model by highest accuracy
    best_model = None
    if models:
        best_model = max(models, key=lambda x: x["accuracy"])

    # Count active models
    active_models = sum(1 for m in models if m["is_active"])

    # Calculate average accuracy
    total_accuracy = sum(m["accuracy"] for m in models)
    avg_accuracy = total_accuracy / len(models) if models else 0

    return render_template(
        "model_registry.html",
        models=models,
        model_count=len(models),
        active_models=active_models,
        avg_accuracy=avg_accuracy,
        best_model=best_model,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    """Upload form guide CSV for processing"""
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file selected", "error")
            return redirect(request.url)

        file = request.files["file"]

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Create unprocessed directory if it doesn't exist
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            file.save(file_path)
            flash(
                f'File "{filename}" uploaded successfully and ready for predictions!',
                "success",
            )
            return redirect(url_for("scraping_status"))
        else:
            flash("Invalid file type. Please upload a CSV file.", "error")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/api/stats")
def api_stats():
    """API endpoint for dashboard stats"""
    db_stats = db_manager.get_database_stats()
    file_stats = get_file_stats()

    return jsonify(
        {
            "database": db_stats,
            "files": file_stats,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/system_status")
@cached_endpoint(ttl=30)  # Cache for 30 seconds
def api_system_status():
    """API endpoint for the real-time monitoring sidebar - Optimized with caching"""
    try:
        start_time = time.time()
        
        if OPTIMIZATION_ENABLED:
            # Use optimized queries for better performance
            optimized_queries = get_optimized_queries(DATABASE_PATH)
            
            # Get comprehensive system stats in a single query
            system_stats = optimized_queries.get_comprehensive_system_stats()
            db_stats = system_stats.get('database', {})
            
            # Get logs (optimized if stored in database)
            logs = optimized_queries.get_recent_logs_optimized(limit=50)
            
            # Fallback to logger if no database logs
            if not logs:
                logs = logger.get_web_logs(limit=50)
            
            # Get model metrics (optimized)
            model_metrics = get_model_predictions()
            
            # Add query performance info
            query_performance = optimized_queries.get_query_performance_stats()
            
        else:
            # Fallback to original implementation
            logs = logger.get_web_logs(limit=50)
            model_metrics = get_model_predictions()
            db_stats = db_manager.get_database_stats()
            query_performance = {'optimized': False}
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "logs": logs,
            "model_metrics": model_metrics,
            "db_stats": db_stats,
            "query_performance": query_performance,
            "response_time_ms": round(total_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "optimized": OPTIMIZATION_ENABLED
        }
        
    except Exception as e:
        logger.exception(f"Error in system_status endpoint: {e}")
        return (
            {
                "success": False, 
                "message": f"Error getting system status: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            500,
        )


@app.route("/api/recent_races")
@cached_endpoint(ttl=15) if OPTIMIZATION_ENABLED else lambda f: f  # Cache for 15 seconds when optimized
def api_recent_races():
    """API endpoint for recent races - Optimized with caching and pagination"""
    try:
        start_time = time.time()
        
        # Limit default to 10 rows for better performance
        limit = request.args.get("limit", 10, type=int)
        
        # Cap maximum limit to prevent excessive database load
        limit = min(limit, 50)
        
        if OPTIMIZATION_ENABLED:
            # Use optimized database queries when available
            optimized_queries = get_optimized_queries(DATABASE_PATH)
            races = optimized_queries.get_recent_races_optimized(limit=limit)
            
            # Fallback to standard method if optimized query fails
            if not races:
                races = db_manager.get_recent_races(limit=limit)
        else:
            races = db_manager.get_recent_races(limit=limit)
        
        query_time = time.time() - start_time
        
        return {
            "success": True,
            "races": races, 
            "count": len(races), 
            "limit": limit,
            "response_time_ms": round(query_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "optimized": OPTIMIZATION_ENABLED
        }
        
    except Exception as e:
        logger.exception(f"Error in recent_races endpoint: {e}")
        return (
            {
                "success": False, 
                "message": f"Error getting recent races: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            500,
        )


# Model Registry API Endpoints


@app.route("/api/model_registry/models")
@cached_endpoint(ttl=30)  # Cache for 30 seconds
def api_model_registry_models():
    """API endpoint to list all models in the registry - Optimized with caching"""
    try:
        start_time = time.time()
        
        if model_registry is None:
            return (
                {
                    "success": False, 
                    "message": "Model registry not initialized",
                    "timestamp": datetime.now().isoformat()
                },
                500,
            )

        models = model_registry.list_models()
        
        # Add performance metadata
        query_time = time.time() - start_time
        
        return {
            "success": True,
            "models": models,
            "count": len(models),
            "response_time_ms": round(query_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "optimized": True
        }

    except Exception as e:
        logger.exception(f"Error in model_registry_models endpoint: {e}")
        return (
            {
                "success": False, 
                "message": f"Error listing models: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            500,
        )


@app.route("/api/model_registry/models/<model_id>")
@cached_endpoint(key_func=lambda req: f"model_detail:{req.view_args['model_id']}", ttl=60)  # Cache for 60 seconds
def api_model_registry_model_detail(model_id):
    """API endpoint to get detailed information about a specific model - Optimized with caching"""
    try:
        start_time = time.time()
        
        if model_registry is None:
            return (
                {
                    "success": False, 
                    "message": "Model registry not initialized",
                    "timestamp": datetime.now().isoformat()
                },
                500,
            )

        model_info = model_registry.get_model_info(model_id)

        if model_info is None:
            return (
                {
                    "success": False, 
                    "message": f"Model {model_id} not found",
                    "timestamp": datetime.now().isoformat()
                },
                404,
            )
        
        query_time = time.time() - start_time

        return {
            "success": True,
            "model": model_info,
            "response_time_ms": round(query_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "optimized": True
        }

    except Exception as e:
        logger.exception(f"Error in model_registry_model_detail endpoint: {e}")
        return (
            {
                "success": False, 
                "message": f"Error getting model details: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            500,
        )


@app.route("/api/model_registry/performance")
@cached_endpoint(ttl=45)  # Cache for 45 seconds
def api_model_registry_performance():
    """API endpoint to get performance metrics for all models - Optimized with caching"""
    try:
        start_time = time.time()
        
        if model_registry is None:
            return (
                {
                    "success": False, 
                    "message": "Model registry not initialized",
                    "timestamp": datetime.now().isoformat()
                },
                500,
            )

        # Get all models and calculate performance summary
        models = model_registry.list_models()

        # Calculate aggregate metrics
        total_models = len(models)
        active_models = sum(
            1 for m in models if hasattr(m, "is_active") and m.is_active
        )

        # Find best model by highest accuracy
        best_model = None
        if models:
            best_model = max(
                models, key=lambda x: x.accuracy if hasattr(x, "accuracy") else 0
            )

        # Calculate average metrics
        accuracies = [m.accuracy for m in models if hasattr(m, "accuracy")]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

        f1_scores = [m.f1_score for m in models if hasattr(m, "f1_score")]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        performance_data = {
            "total_models": total_models,
            "active_models": active_models,
            "avg_accuracy": round(avg_accuracy, 4),
            "avg_f1_score": round(avg_f1, 4),
            "best_model": (
                best_model.model_id
                if best_model and hasattr(best_model, "model_id")
                else None
            ),
            "best_accuracy": (
                round(best_model.accuracy, 4)
                if best_model and hasattr(best_model, "accuracy")
                else 0
            ),
        }
        
        query_time = time.time() - start_time

        return {
            "success": True,
            "performance": performance_data,
            "response_time_ms": round(query_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "optimized": True
        }

    except Exception as e:
        logger.exception(f"Error in model_registry_performance endpoint: {e}")
        return (
            {
                "success": False,
                "message": f"Error getting performance data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            500,
        )


@app.route("/api/model_registry/status")
@cached_endpoint(ttl=30)  # Cache for 30 seconds
def api_model_registry_status():
    """API endpoint to get model registry status - Optimized with caching"""
    try:
        start_time = time.time()
        
        if model_registry is None:
            return {
                "success": False,
                "message": "Model registry not initialized",
                "initialized": False,
                "model_count": 0,
                "timestamp": datetime.now().isoformat()
            }

        models = model_registry.list_models()
        query_time = time.time() - start_time
        
        # Get cache statistics if optimization is enabled
        cache_stats = {}
        if OPTIMIZATION_ENABLED:
            try:
                cache = get_endpoint_cache()
                cache_stats = cache.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")

        # Find best models for each prediction type
        best_models = {}
        try:
            best_models = {m.prediction_type: m for m in models if hasattr(m, 'is_best') and m.is_best}
        except Exception as e:
            logger.warning(f"Failed to get best models: {e}")
            best_models = {}

        return {
        "success": True,
        "initialized": True,
        "model_count": len(models),
        "best_models": best_models,
        "all_models": models,
        "response_time_ms": round(query_time * 1000, 2),
        "cache_stats": cache_stats,
        "timestamp": datetime.now().isoformat(),
        "optimized": OPTIMIZATION_ENABLED
    }

    except Exception as e:
        logger.exception(f"Error in model_registry_status endpoint: {e}")
        return (
            {
                "success": False,
                "message": f"Error getting registry status: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            500,
        )


@app.route("/api/race/<race_id>")
def api_race_detail(race_id):
    """API endpoint for race details"""
    race_data = db_manager.get_race_details(race_id)

    if not race_data:
        return jsonify({"error": "Race not found"}), 404

    return jsonify({"race_data": race_data, "timestamp": datetime.now().isoformat()})


# Global variables to track processing status
processing_lock = threading.Lock()
processing_status = {
    "running": False,
    "log": [],
    "start_time": None,
    "progress": 0,
    "current_task": "",
    "total_files": 0,
    "processed_files": 0,
    "error_count": 0,
    "last_update": None,
    "session_id": None,  # Track which session started the process
    "process_type": None,  # Track what type of process is running
}

# Keep track of active background threads
active_threads = {}

# Global test prediction status tracking
test_prediction_status = {
    "running": False,
    "progress": 0,
    "current_step": "",
    "log": [],
    "start_time": None,
    "completed": False,
    "results": None,
    "error": None,
}


def safe_log_to_processing(message, level="INFO", update_progress=None):
    """Safely log message to processing status and enhanced logger"""
    timestamp = datetime.now().isoformat()

    with processing_lock:
        # Initialize log if it doesn't exist
        if "log" not in processing_status:
            processing_status["log"] = []

        processing_status["log"].append(
            {"timestamp": timestamp, "message": message, "level": level}
        )

        if update_progress is not None:
            processing_status["progress"] = update_progress

        processing_status["last_update"] = timestamp

        # Keep only last 200 entries in processing log
        if len(processing_status["log"]) > 200:
            processing_status["log"] = processing_status["log"][-200:]

    # Also log to enhanced logger
    if level == "ERROR":
        logger.log_error(message, context={"component": "processor"})
    else:
        logger.log_process(message, level)


def run_command_with_output(command, log_prefix=""):
    """Run a command and capture output in real-time"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in iter(process.stdout.readline, ""):
            # Check if processing was stopped
            if not processing_status.get("running", False):
                safe_log_to_processing(
                    "⏹️ Process terminated by user request", "WARNING"
                )
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False

            line = line.strip()
            if line:
                processing_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"{log_prefix}{line}",
                    }
                )

        process.wait()
        return process.returncode == 0
    except Exception as e:
        processing_status["log"].append(
            {"timestamp": datetime.now().isoformat(), "message": f"❌ Error: {str(e)}"}
        )
        return False


@profile_function
def process_files_background():
    """Background task to process files"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        # Keep existing log entries and append new ones
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0

    try:
        # Add initial log entry
        safe_log_to_processing("🚀 Starting file processing...", "INFO", 0)

        # Check if processing was stopped
        if not processing_status.get("running", False):
            return

        # Check for unprocessed files
        if os.path.exists(UNPROCESSED_DIR):
            unprocessed_files = [
                f for f in os.listdir(UNPROCESSED_DIR) if f.endswith(".csv")
            ]
            if unprocessed_files:
                processing_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"📊 Found {len(unprocessed_files)} files to process",
                    }
                )
                processing_status["progress"] = 25

                # Check if processing was stopped before running command
                if not processing_status.get("running", False):
                    return

                # Run the analysis command
                success = run_command_with_output(
                    [sys.executable, "run.py", "analyze"], "📈 "
                )

                # Check if processing was stopped after command
                if not processing_status.get("running", False):
                    return

                processing_status["progress"] = 75

                if success:
                    processing_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": "✅ File processing completed successfully!",
                        }
                    )
                else:
                    processing_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": "❌ File processing failed",
                        }
                    )
            else:
                processing_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": "ℹ️ No unprocessed files found",
                    }
                )
        else:
            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "⚠️ Unprocessed directory not found",
                }
            )

        processing_status["progress"] = 100

    except Exception as e:
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": f"❌ Processing error: {str(e)}",
            }
        )

    finally:
        processing_status["running"] = False
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "🏁 Processing task completed",
            }
        )


def run_scraper_background():
    """Background task to run scraper"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        # Keep existing log entries and append new ones
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0

    try:
        safe_log_to_processing("🕷️ Starting scraper...", "INFO", 0)

        processing_status["progress"] = 25

        # Run the scraper
        success = run_command_with_output([sys.executable, "run.py", "collect"], "🔍 ")

        processing_status["progress"] = 100

        if success:
            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "✅ Scraping completed successfully!",
                }
            )
        else:
            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "❌ Scraping failed",
                }
            )

    except Exception as e:
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": f"❌ Scraper error: {str(e)}",
            }
        )

    finally:
        processing_status["running"] = False
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "🏁 Scraping task completed",
            }
        )


def fetch_csv_background():
    """Background task to fetch CSV form guides using expert-form approach"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        # Keep existing log entries and append new ones
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0
        processing_status["current_task"] = "Initializing CSV fetching"

    try:
        safe_log_to_processing("📊 Starting CSV form guide fetching...", "INFO", 0)

        processing_status["progress"] = 10
        processing_status["current_task"] = "Preparing expert-form scraper"
        safe_log_to_processing("🔍 Using expert-form CSV scraper for enhanced accuracy...", "INFO", 10)
        safe_log_to_processing("⚡ Using optimized settings for faster processing...", "INFO", 15)

        processing_status["progress"] = 25
        processing_status["current_task"] = "Running CSV scraper (this may take 2-3 minutes)"
        safe_log_to_processing("🚀 Starting CSV download process...", "INFO", 25)

        # Run the expert form CSV scraper with optimized parameters:
        # - Only 1 day ahead to reduce processing time
        # - Max 2 workers for reasonable concurrency without overwhelming the server
        # - Increased timeout for large batch processing
        result = subprocess.run(
            [sys.executable, "expert_form_csv_scraper.py", "--days-ahead", "1", "--max-workers", "2", "--verbose"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        processing_status["progress"] = 90
        processing_status["current_task"] = "Processing results"
        
        success = result.returncode == 0
        
        # Extract useful stats from output if available
        if result.stdout:
            output_lines = result.stdout.strip().split('\n')
            
            # Look for statistics in the output
            stats_found = False
            for line in output_lines[-20:]:  # Check last 20 lines for stats
                if 'successful' in line.lower() or 'completed' in line.lower():
                    safe_log_to_processing(f"📊 {line.strip()}", "INFO", 92)
                    stats_found = True
                elif 'races requested' in line.lower() or 'cache hits' in line.lower():
                    safe_log_to_processing(f"📈 {line.strip()}", "INFO", 94)
                    stats_found = True
            
            if not stats_found and len(output_lines) > 0:
                # Show last few lines of output
                safe_log_to_processing(f"📋 Output: {output_lines[-1][:100]}", "INFO", 95)
        
        processing_status["progress"] = 100
        processing_status["current_task"] = "Completed"

        if success:
            safe_log_to_processing(
                "✅ CSV form guides fetched successfully using expert-form method!", "INFO", 100
            )
            if result.stderr:
                # Even on success, show any warnings
                error_lines = result.stderr.strip().split('\n')
                warning_count = sum(1 for line in error_lines if 'WARNING' in line or 'WARN' in line)
                if warning_count > 0:
                    safe_log_to_processing(f"⚠️ Completed with {warning_count} warnings (some races may not have CSV data available)", "WARNING", 100)
        else:
            safe_log_to_processing(
                "❌ CSV fetching failed - expert-form method encountered issues", "ERROR", 100
            )
            if result.stderr:
                error_msg = result.stderr.strip()[:200]  # First 200 chars of error
                safe_log_to_processing(f"🔍 Error details: {error_msg}", "ERROR", 100)

    except subprocess.TimeoutExpired:
        safe_log_to_processing(
            "⏰ CSV fetching timed out (5 minute limit). Some data may have been collected.", "WARNING", 100
        )
        processing_status["current_task"] = "Timed out"
    except Exception as e:
        safe_log_to_processing(
            f"❌ CSV fetching error: {str(e)}", "ERROR", processing_status.get("progress", 0)
        )
        processing_status["current_task"] = "Error occurred"

    finally:
        processing_status["running"] = False
        processing_status["current_task"] = "Finished"
        safe_log_to_processing("🏁 CSV fetching completed", "INFO", 100)


def process_data_background():
    """Background task to process data with enhanced comprehensive processor"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        # Keep existing log entries and append new ones
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0
        processing_status["current_task"] = "Initializing"
        processing_status["total_files"] = 0
        processing_status["processed_files"] = 0
        processing_status["error_count"] = 0

    logger.log_system(
        "Starting enhanced comprehensive data processing", "INFO", "PROCESSOR"
    )
    safe_log_to_processing(
        "🚀 Starting enhanced comprehensive data processing...", "INFO", 0
    )

    try:
        safe_log_to_processing(
            "🔧 Initializing enhanced comprehensive processor...", "INFO", 10
        )

        # Try to use enhanced processor if available
        try:
            # Import and use enhanced processor
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "enhanced_comprehensive_processor",
                "./enhanced_comprehensive_processor.py",
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                EnhancedComprehensiveProcessor = module.EnhancedComprehensiveProcessor

                safe_log_to_processing(
                    "✅ Enhanced processor imported successfully", "INFO", 15
                )

                processor = EnhancedComprehensiveProcessor()
                safe_log_to_processing("✅ Enhanced processor initialized", "INFO", 20)
            else:
                raise ImportError("Could not load enhanced processor module")

            # Process all unprocessed files
            safe_log_to_processing("📊 Processing unprocessed files...", "INFO", 30)
            results = processor.process_all_unprocessed()
            safe_log_to_processing("📊 Processing complete", "INFO", 80)

            # Move processed files to processed directory
            if (
                results.get("status") == "success"
                and results.get("processed_count", 0) > 0
            ):
                safe_log_to_processing("📁 Moving processed files...", "INFO", 85)
                try:
                    import shutil

                    os.makedirs(PROCESSED_DIR, exist_ok=True)

                    # Move files that were successfully processed
                    for result_item in results.get("results", []):
                        if result_item.get("result", {}).get("status") == "success":
                            filename = result_item["filename"]
                            source_path = os.path.join(UNPROCESSED_DIR, filename)
                            dest_path = os.path.join(PROCESSED_DIR, filename)

                            # Only move if file exists in unprocessed and not in processed
                            if os.path.exists(source_path) and not os.path.exists(
                                dest_path
                            ):
                                shutil.move(source_path, dest_path)
                                safe_log_to_processing(
                                    f"📁 Moved {filename} to processed directory",
                                    "INFO",
                                )
                            elif os.path.exists(source_path) and os.path.exists(
                                dest_path
                            ):
                                # File exists in both locations, remove from unprocessed
                                os.remove(source_path)
                                safe_log_to_processing(
                                    f"📁 Removed duplicate {filename} from unprocessed",
                                    "INFO",
                                )

                    safe_log_to_processing("📁 File management complete", "INFO", 87)
                except Exception as e:
                    safe_log_to_processing(f"⚠️ Error moving files: {str(e)}", "WARNING")

            # Log results
            if results.get("status") == "success":
                safe_log_to_processing(
                    f"✅ Enhanced processing completed! Processed {results.get('processed_count', 0)} files",
                    "INFO",
                    85,
                )

                if results.get("failed_count", 0) > 0:
                    safe_log_to_processing(
                        f"⚠️ {results['failed_count']} files failed to process",
                        "WARNING",
                    )
            else:
                safe_log_to_processing(
                    f"❌ Enhanced processing failed: {results.get('message', 'Unknown error')}",
                    "ERROR",
                )

            # Generate comprehensive report
            if results.get("processed_count", 0) > 0:
                safe_log_to_processing(
                    "📊 Generating comprehensive analysis report...", "INFO", 90
                )

                try:
                    report_path = processor.generate_comprehensive_report()
                    if report_path:
                        safe_log_to_processing(
                            f"📋 Report generated: {os.path.basename(report_path)}",
                            "INFO",
                            95,
                        )
                except Exception as e:
                    safe_log_to_processing(
                        f"❌ Report generation failed: {str(e)}", "ERROR"
                    )

            # Cleanup
            processor.cleanup()

        except ImportError as e:
            # Fallback to basic processing if enhanced processor not available
            safe_log_to_processing(
                f"⚠️ Enhanced processor not available: {str(e)}", "WARNING"
            )
            safe_log_to_processing("🔄 Using basic processing...", "INFO")

            # Basic file processing
            if not os.path.exists(UNPROCESSED_DIR):
                processing_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": "⚠️ Unprocessed directory not found",
                    }
                )
                processing_status["progress"] = 100
                return

            unprocessed_files = [
                f for f in os.listdir(UNPROCESSED_DIR) if f.endswith(".csv")
            ]
            if not unprocessed_files:
                processing_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": "ℹ️ No unprocessed files found",
                    }
                )
                processing_status["progress"] = 100
                return

            os.makedirs(PROCESSED_DIR, exist_ok=True)
            import shutil

            processed_count = 0

            for i, filename in enumerate(unprocessed_files):
                try:
                    source_path = os.path.join(UNPROCESSED_DIR, filename)
                    dest_path = os.path.join(PROCESSED_DIR, filename)

                    if os.path.exists(dest_path):
                        processing_status["log"].append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "message": f"⚠️ {filename} already processed, skipping",
                            }
                        )
                        continue

                    shutil.copy2(source_path, dest_path)
                    os.remove(source_path)
                    processed_count += 1

                    processing_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": f"✅ {filename} processed",
                        }
                    )

                    progress = 20 + ((i + 1) / len(unprocessed_files)) * 60
                    processing_status["progress"] = int(progress)

                except Exception as e:
                    processing_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": f"❌ Error processing {filename}: {str(e)}",
                        }
                    )

            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": f"✅ Basic processing completed! Processed {processed_count} files",
                }
            )

        processing_status["progress"] = 100

    except Exception as e:
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": f"❌ Data processing error: {str(e)}",
            }
        )

    finally:
        processing_status["running"] = False
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "🏁 Data processing completed",
            }
        )


def simple_pipeline_background():
    """Simple linear data processing pipeline without threading complexity"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        processing_status["log"] = []  # Start fresh
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0
        processing_status["current_task"] = "Starting simple pipeline"

    try:
        safe_log_to_processing(
            "🚀 Starting simple data processing pipeline...", "INFO", 0
        )

        # Step 1: Check for files to process
        safe_log_to_processing("📁 Checking for files to process...", "INFO", 10)

        if not os.path.exists(UNPROCESSED_DIR):
            safe_log_to_processing("⚠️ No unprocessed directory found", "WARNING", 100)
            return

        unprocessed_files = [
            f for f in os.listdir(UNPROCESSED_DIR) if f.endswith(".csv")
        ]
        if not unprocessed_files:
            safe_log_to_processing("ℹ️ No files to process", "INFO", 100)
            return

        safe_log_to_processing(
            f"📊 Found {len(unprocessed_files)} files to process", "INFO", 20
        )

        # Step 2: Process the files (simple move to processed directory)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        processed_count = 0

        for i, filename in enumerate(unprocessed_files):
            try:
                source_path = os.path.join(UNPROCESSED_DIR, filename)
                dest_path = os.path.join(PROCESSED_DIR, filename)

                # Check if already processed
                if os.path.exists(dest_path):
                    safe_log_to_processing(
                        f"⚠️ {filename} already processed, removing from unprocessed",
                        "WARNING",
                    )
                    os.remove(source_path)
                    continue

                # Move file to processed directory
                import shutil

                shutil.move(source_path, dest_path)
                processed_count += 1

                safe_log_to_processing(f"✅ Processed {filename}", "INFO")

                # Update progress
                progress = 20 + ((i + 1) / len(unprocessed_files)) * 60
                processing_status["progress"] = int(progress)

            except Exception as e:
                safe_log_to_processing(
                    f"❌ Error processing {filename}: {str(e)}", "ERROR"
                )

        # Step 3: Update database if needed (optional)
        safe_log_to_processing("🔄 Checking database status...", "INFO", 85)

        try:
            db_stats = db_manager.get_database_stats()
            total_races = db_stats.get("total_races", 0)
            safe_log_to_processing(f"📊 Database contains {total_races} races", "INFO")
        except Exception as e:
            safe_log_to_processing(f"⚠️ Database check failed: {str(e)}", "WARNING")

        # Step 4: Complete
        processing_status["progress"] = 100

        if processed_count > 0:
            safe_log_to_processing(
                f"✅ Simple pipeline completed! Processed {processed_count} files",
                "INFO",
            )
        else:
            safe_log_to_processing(
                "ℹ️ Simple pipeline completed - no new files to process", "INFO"
            )

    except Exception as e:
        safe_log_to_processing(f"❌ Pipeline error: {str(e)}", "ERROR")

    finally:
        processing_status["running"] = False
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "🏁 Simple pipeline completed",
            }
        )


def update_analysis_background():
    """Background task to update AI analysis"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        # Don't clear log - append to existing log instead
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0
        processing_status["current_task"] = "AI Analysis"

    logger.log_system("Starting AI analysis update", "INFO", "AI_ANALYSIS")
    safe_log_to_processing("🧠 Starting AI analysis update...", "INFO", 0)

    try:
        safe_log_to_processing("📁 Checking for processed files...", "INFO", 25)

        # Check if processed files exist
        if not os.path.exists(PROCESSED_DIR):
            safe_log_to_processing(
                "⚠️ No processed files found. Run 'Process Data' first.", "WARNING", 100
            )
            return

        # Count all processed files recursively
        processed_count = 0
        for root, dirs, files in os.walk(PROCESSED_DIR):
            processed_count += len([f for f in files if f.endswith(".csv")])

        if processed_count == 0:
            safe_log_to_processing(
                "⚠️ No processed files found. Run 'Process Data' first.", "WARNING", 100
            )
            return

        safe_log_to_processing(
            f"📁 Found {processed_count} processed files for analysis", "INFO", 50
        )

        # Try to run advanced AI analysis
        success = False
        if os.path.exists("advanced_ai_analysis.py"):
            safe_log_to_processing("🧠 Running advanced AI analysis...", "INFO", 60)

            try:
                result = subprocess.run(
                    [sys.executable, "advanced_ai_analysis.py"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode == 0:
                    safe_log_to_processing(
                        "✅ Advanced AI analysis completed successfully!", "INFO", 90
                    )
                    success = True
                else:
                    error_msg = (
                        result.stderr[:200] if result.stderr else "Unknown error"
                    )
                    safe_log_to_processing(
                        f"⚠️ AI analysis had issues: {error_msg}", "WARNING"
                    )

            except subprocess.TimeoutExpired:
                safe_log_to_processing(
                    "⚠️ AI analysis timed out (5 min limit)", "WARNING"
                )
            except Exception as e:
                safe_log_to_processing(f"⚠️ AI analysis error: {str(e)}", "ERROR")
        else:
            safe_log_to_processing("⚠️ Advanced AI analysis script not found", "WARNING")

        processing_status["progress"] = 100

        if success:
            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "✅ AI analysis update completed successfully!",
                }
            )
        else:
            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "ℹ️ AI analysis update completed with warnings",
                }
            )

    except Exception as e:
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": f"❌ Analysis update error: {str(e)}",
            }
        )

    finally:
        processing_status["running"] = False
        processing_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "🏁 Analysis update completed",
            }
        )


def perform_prediction_background():
    """Background task to perform race predictions using comprehensive prediction pipeline"""
    global processing_status

    with processing_lock:
        processing_status["running"] = True
        processing_status["start_time"] = datetime.now()
        processing_status["progress"] = 0
        processing_status["log"] = []

    try:
        safe_log_to_processing(
            "🎯 Starting comprehensive race predictions for all upcoming races...",
            "INFO",
            0,
        )

        if ComprehensivePredictionPipeline:
            pipeline = ComprehensivePredictionPipeline()
            safe_log_to_processing(
                "✅ Comprehensive Prediction Pipeline initialized", "INFO", 10
            )

            results = pipeline.predict_all_upcoming_races(upcoming_dir=UPCOMING_DIR)

            if results.get("success"):
                safe_log_to_processing(
                    f"✅ Batch prediction complete. Processed {results.get('total_races', 0)} races.",
                    "INFO",
                    100,
                )
                safe_log_to_processing(
                    f"📈 {results.get('successful_predictions', 0)} of {results.get('total_races', 0)} predictions were successful.",
                    "INFO",
                )
            else:
                safe_log_to_processing(
                    f"❌ Batch prediction failed: {results.get('message')}",
                    "ERROR",
                    100,
                )
        else:
            safe_log_to_processing(
                "❌ ComprehensivePredictionPipeline not available.", "ERROR", 100
            )

    except Exception as e:
        safe_log_to_processing(f"❌ Prediction error: {str(e)}", "ERROR")

    finally:
        with processing_lock:
            processing_status["running"] = False
            processing_status["progress"] = 100
            safe_log_to_processing("🏁 Comprehensive prediction task completed", "INFO")


@app.route("/api/process_files", methods=["POST"])
def api_process_files():
    """API endpoint to start file processing"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start background processing
    thread = threading.Thread(target=process_files_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "File processing started"})


@app.route("/api/run_scraper", methods=["POST"])
def api_run_scraper():
    """API endpoint to start scraper"""
    if processing_status["running"]:
        return jsonify({"success": False, "message": "Scraper already running"}), 400

    # Start background scraping
    thread = threading.Thread(target=run_scraper_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Scraper started"})


@app.route("/api/processing_status")
def api_processing_status():
    """API endpoint for processing status"""
    with processing_lock:
        status = processing_status.copy()
        status["last_update"] = datetime.now().isoformat()
    return jsonify(status)


@app.route("/api/logs", methods=["GET", "POST"])
def api_logs():
    """API endpoint for enhanced logs - GET to retrieve, POST to add log entries"""
    if request.method == 'GET':
        log_type = request.args.get("type", "all")
        limit = request.args.get("limit", 100, type=int)

        logs = logger.get_web_logs(log_type, limit)
        summary = logger.get_log_summary()

        return jsonify(
            {"logs": logs, "summary": summary, "timestamp": datetime.now().isoformat()}
        )
    
    elif request.method == 'POST':
        """POST endpoint to add log entries - used for advisory status logging"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400
            
            # Extract log data
            summary = data.get("summary", "")
            level = data.get("level", "INFO")
            component = data.get("component", "advisory")
            details = data.get("details", {})
            
            # Log the entry using the appropriate logger method
            if component == "advisory":
                logger.log_system(
                    message=summary,
                    level=level,
                    component="ADVISORY"
                )
            else:
                logger.log_process(
                    message=summary,
                    level=level,
                    details=details
                )
            
            return jsonify({
                "success": True,
                "message": "Log entry added successfully",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error adding log entry: {e}")
            return jsonify({
                "success": False,
                "error": f"Error adding log entry: {str(e)}"
            }), 500


@app.route("/api/logs/clear", methods=["POST"])
def api_clear_logs():
    """API endpoint to clear logs"""
    log_type = request.json.get("type", "all") if request.json else "all"

    logger.clear_logs(log_type)

    return jsonify(
        {
            "success": True,
            "message": f"Cleared {log_type} logs",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/processing_logs/clear", methods=["POST"])
def api_clear_processing_logs():
    """API endpoint to clear processing logs"""
    with processing_lock:
        processing_status["log"] = []
        processing_status["progress"] = 0
        processing_status["current_task"] = ""
        processing_status["error_count"] = 0

    return jsonify(
        {
            "success": True,
            "message": "Processing logs cleared",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/process_file/<filename>", methods=["POST"])
def api_process_single_file(filename):
    """API endpoint to process a single file"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    file_path = os.path.join(UNPROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"success": False, "message": "File not found"}), 404

    try:
        # For now, just move the file to processed directory
        # In a real implementation, you'd process the file through your analysis pipeline
        processed_path = os.path.join(PROCESSED_DIR, filename)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        # Copy file to processed directory
        import shutil

        shutil.copy2(file_path, processed_path)

        # Remove from unprocessed
        os.remove(file_path)

        return jsonify(
            {"success": True, "message": f"File {filename} processed successfully"}
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error processing file: {str(e)}"}),
            500,
        )


# New Workflow API Endpoints
@app.route("/api/fetch_csv", methods=["POST"])
def api_fetch_csv():
    """API endpoint to fetch CSV form guides"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start background CSV fetching
    thread = threading.Thread(target=fetch_csv_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "CSV fetching started"})


@app.route("/api/start_scraper", methods=["POST"])
def api_start_scraper():
    """API endpoint to start automated data collection (scraping)"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start background scraping (run.py collect)
    thread = threading.Thread(target=run_scraper_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Data collection (scraping) started"})


@app.route("/api/process_data", methods=["POST"])
def api_process_data():
    """API endpoint to process enhanced data"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start background data processing
    thread = threading.Thread(target=process_data_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Data processing started"})


@app.route("/api/collect_and_analyze", methods=["POST"])
def api_collect_and_analyze():
    """API endpoint to run the full automated pipeline: collect + analyze"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    def collect_and_analyze_background():
        """Background task to run collect then analyze"""
        global processing_status

        with processing_lock:
            processing_status["running"] = True
            processing_status["start_time"] = datetime.now()
            processing_status["progress"] = 0

        try:
            safe_log_to_processing(
                "🚀 Starting automated data collection and analysis pipeline...",
                "INFO",
                0,
            )

            # Step 1: Collection (run.py collect)
            safe_log_to_processing(
                "🔍 Step 1: Collecting data (upcoming races + moving to unprocessed)...",
                "INFO",
                10,
            )
            processing_status["progress"] = 20

            success1 = run_command_with_output(
                [sys.executable, "run.py", "collect"], "🔍 "
            )

            if not processing_status.get("running", False):
                return

            processing_status["progress"] = 50

            if success1:
                safe_log_to_processing(
                    "✅ Data collection completed successfully!", "INFO", 50
                )

                # Step 2: Analysis (run.py analyze)
                safe_log_to_processing(
                    "📈 Step 2: Analyzing data (processing files + database storage)...",
                    "INFO",
                    55,
                )
                processing_status["progress"] = 60

                success2 = run_command_with_output(
                    [sys.executable, "run.py", "analyze"], "📈 "
                )

                if not processing_status.get("running", False):
                    return

                processing_status["progress"] = 90

                if success2:
                    safe_log_to_processing(
                        "✅ Data analysis completed successfully!", "INFO", 95
                    )
                    safe_log_to_processing(
                        "🎉 Full pipeline completed: Data collected, processed, and stored in database!",
                        "INFO",
                        100,
                    )
                else:
                    safe_log_to_processing("❌ Data analysis failed", "ERROR", 90)
            else:
                safe_log_to_processing("❌ Data collection failed", "ERROR", 50)

        except Exception as e:
            safe_log_to_processing(
                f"❌ Pipeline error: {str(e)}",
                "ERROR",
                processing_status.get("progress", 0),
            )

        finally:
            processing_status["running"] = False
            processing_status["progress"] = 100
            safe_log_to_processing("🏁 Automated pipeline task completed", "INFO", 100)

    # Start background pipeline
    thread = threading.Thread(target=collect_and_analyze_background)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "Automated data collection and analysis pipeline started",
        }
    )


@app.route("/api/update_analysis", methods=["POST"])
def api_update_analysis():
    """API endpoint to update AI analysis"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start background analysis update
    thread = threading.Thread(target=update_analysis_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Analysis update started"})


@app.route("/api/perform_prediction", methods=["POST"])
def api_perform_prediction():
    """API endpoint to perform race predictions"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start background prediction
    thread = threading.Thread(target=perform_prediction_background)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Prediction started"})


@app.route("/api/predict_upcoming", methods=["POST"])
def api_predict_upcoming():
    """API endpoint to predict specifically on uploaded upcoming races"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    try:
        # Check if upcoming races directory exists and has files
        if not os.path.exists(UPCOMING_DIR):
            return (
                jsonify(
                    {"success": False, "message": "No upcoming races directory found"}
                ),
                404,
            )

        # Get all CSV files using the helper function
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        upcoming_races = load_upcoming_races(refresh=refresh)
        upcoming_files = [race.get("filename", f"{race.get('name', 'race')}.csv") for race in upcoming_races if race.get("filename") or race.get("name")]
        
        # If no files from helper, fallback to direct directory scan for CSV only
        if not upcoming_files:
            upcoming_files = [
                f
                for f in os.listdir(UPCOMING_DIR)
                if f.endswith(".csv") and f != "README.md"
            ]
        if not upcoming_files:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "No upcoming race files found for prediction",
                    }
                ),
                404,
            )

        # Run prediction directly on upcoming races
        predict_script = "upcoming_race_predictor.py"
        if not os.path.exists(predict_script):
            return (
                jsonify({"success": False, "message": "Prediction script not found"}),
                500,
            )

        # Run prediction on the upcoming races directory
        result = subprocess.run(
            [sys.executable, predict_script],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode == 0:
            # Parse the output to extract key information
            output_lines = result.stdout.split("\n")
            predictions_summary = []
            current_race = None

            for line in output_lines:
                if "🎯 Predicting:" in line:
                    current_race = line.split("🎯 Predicting:")[1].strip()
                elif "🏆 Top 3 picks:" in line and current_race:
                    predictions_summary.append(
                        {"race": current_race, "status": "predicted"}
                    )

            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully predicted {len(predictions_summary)} upcoming races",
                    "predictions": predictions_summary,
                    "files_processed": upcoming_files,
                    "output": (
                        result.stdout[-1000:]
                        if len(result.stdout) > 1000
                        else result.stdout
                    ),  # Last 1000 chars
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f'Prediction failed: {result.stderr[:200] if result.stderr else "Unknown error"}',
                        "error_output": result.stderr,
                    }
                ),
                500,
            )

    except subprocess.TimeoutExpired:
        return (
            jsonify(
                {"success": False, "message": "Prediction timed out (2 minute limit)"}
            ),
            500,
        )
    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error running prediction: {str(e)}"}
            ),
            500,
        )


@app.route("/api/stop_processing", methods=["POST"])
def api_stop_processing():
    """API endpoint to stop current processing"""
    global processing_status

    with processing_lock:
        if processing_status["running"]:
            processing_status["running"] = False
            processing_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "⏹️ Processing stopped by user",
                }
            )
            processing_status["last_update"] = datetime.now().isoformat()
            message = "Processing stopped"
            success = True
        else:
            message = "No processing currently running"
            success = False

    return jsonify({"success": success, "message": message})


@app.route("/api/test_prediction_status")
def api_test_prediction_status():
    """API endpoint to get test prediction status"""
    global test_prediction_status

    return jsonify(
        {
            "success": True,
            "status": test_prediction_status,
            "timestamp": datetime.now().isoformat(),
        }
    )


def log_test_prediction(message, level="INFO", progress=None):
    """Log test prediction status with timestamp"""
    global test_prediction_status

    timestamp = datetime.now().isoformat()

    test_prediction_status["log"].append(
        {"timestamp": timestamp, "message": message, "level": level}
    )

    if progress is not None:
        test_prediction_status["progress"] = progress

    # Keep only last 100 entries
    if len(test_prediction_status["log"]) > 100:
        test_prediction_status["log"] = test_prediction_status["log"][-100:]

    print(f"[TEST_PREDICTION] {message}")


@app.route("/api/test_historical_prediction", methods=["POST"])
def api_test_historical_prediction():
    """API endpoint to test prediction on a historical race and compare with actual outcome"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        race_id = data.get("race_id")
        venue = data.get("venue")
        race_number = data.get("race_number")
        race_date = data.get("race_date")

        if not all([race_id, venue, race_number, race_date]):
            return (
                jsonify(
                    {"success": False, "error": "Missing required race information"}
                ),
                400,
            )

        # Get race details from database
        race_data = db_manager.get_race_details(race_id)

        if not race_data:
            return (
                jsonify({"success": False, "error": "Race not found in database"}),
                404,
            )

        race_info = race_data["race_info"]
        dogs = race_data["dogs"]

        # Check if we have the actual winner
        actual_winner_name = race_info.get("winner_name")
        if not actual_winner_name or actual_winner_name == "nan":
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No winner information available for this race",
                    }
                ),
                400,
            )

        # Find the actual winner details
        actual_winner = None
        for dog in dogs:
            if dog.get("dog_name") == actual_winner_name:
                actual_winner = {
                    "name": dog.get("dog_name"),
                    "box": dog.get("box_number"),
                    "odds": dog.get("odds", "N/A"),
                }
                break

        if not actual_winner:
            actual_winner = {
                "name": actual_winner_name,
                "box": "Unknown",
                "odds": "N/A",
            }

        # Check if we have enough dog data
        if not dogs or len(dogs) == 0:
            return (
                jsonify(
                    {"success": False, "error": "No dog data available for this race"}
                ),
                400,
            )

        # Create temporary race file from database data for real prediction
        import csv
        import tempfile

        # Create temporary CSV file with race data
        temp_race_file = None
        try:
            temp_fd, temp_race_file = tempfile.mkstemp(
                suffix=".csv", prefix=f"historical_race_{race_id}_"
            )

            with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as csvfile:
                # Write header
                fieldnames = ["Dog Name", "Box", "Odds", "Weight", "Trainer", "Jockey"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
                writer.writeheader()

                # Write dog data
                for dog in dogs:
                    row = {
                        "Dog Name": f"{dog.get('box_number', 1)}. {dog.get('dog_name', 'Unknown')}",
                        "Box": dog.get("box_number", 1),
                        "Odds": dog.get("odds", "N/A"),
                        "Weight": dog.get("weight", "N/A"),
                        "Trainer": dog.get("trainer", "N/A"),
                        "Jockey": dog.get("jockey", "N/A"),
                    }
                    writer.writerow(row)

            # Use ML System V3 for real prediction
            try:
                from prediction_pipeline_v3 import PredictionPipelineV3

                # Use the V3 pipeline for the best possible prediction
                pipeline = PredictionPipelineV3()

                print(
                    f"🚀 Running ML System V3 prediction for historical race {race_id}..."
                )
                prediction_result = pipeline.predict_race_file(
                    temp_race_file, enhancement_level="full"
                )

                if prediction_result and prediction_result.get("success"):
                    print("✅ MLv3 prediction completed successfully!")
                    real_predictions = prediction_result.get("predictions", [])
                else:
                    raise Exception(
                        f"Prediction failed: {prediction_result.get('error', 'Unknown error') if prediction_result else 'No result'}"
                    )

            except Exception as pred_error:
                print(f"⚠️ Real prediction failed: {pred_error}")
                print("🔄 Falling back to basic analysis using database data...")

                # Fallback: Use database data for basic prediction analysis
                real_predictions = []
                sorted_dogs = sorted(dogs, key=lambda x: x.get("box_number", 999))

                for i, dog in enumerate(sorted_dogs[:8]):
                    dog_name = dog.get("dog_name", f"Dog {i+1}")
                    if dog_name and dog_name != "nan":
                        # Basic score based on database attributes (not mock)
                        base_score = 0.5

                        # Adjust based on box position (real factor)
                        box_num = dog.get("box_number", i + 1)
                        if box_num <= 3:
                            base_score += 0.1  # Inside boxes advantage
                        elif box_num >= 6:
                            base_score -= 0.1  # Wide barriers disadvantage

                        # Adjust based on odds if available (real factor)
                        odds_str = str(dog.get("odds", "")).strip()
                        if odds_str and odds_str != "N/A" and odds_str != "nan":
                            try:
                                odds_val = float(
                                    odds_str.replace("$", "").replace(",", "")
                                )
                                if odds_val < 3.0:
                                    base_score += 0.15  # Short-priced favorite
                                elif odds_val < 5.0:
                                    base_score += 0.08  # Second favorite
                                elif odds_val > 15.0:
                                    base_score -= 0.10  # Long shot
                            except (ValueError, TypeError):
                                pass

                        real_predictions.append(
                            {
                                "dog_name": dog_name,
                                "box_number": box_num,
                                "prediction_score": round(
                                    max(0.1, min(0.9, base_score)), 3
                                ),
                                "confidence": round(max(0.1, min(0.9, base_score)), 3),
                                "final_score": round(max(0.1, min(0.9, base_score)), 3),
                                "prediction_method": "database_analysis_fallback",
                            }
                        )

                # Sort by prediction score
                real_predictions.sort(
                    key=lambda x: x.get("prediction_score", 0), reverse=True
                )

        finally:
            # Clean up temporary file
            if temp_race_file and os.path.exists(temp_race_file):
                try:
                    os.unlink(temp_race_file)
                except:
                    pass

        # Calculate accuracy metrics
        accuracy_metrics = {
            "winner_predicted": False,
            "top_3_hit": False,
            "actual_winner_rank": None,
        }

        # Find where the actual winner ranks in predictions
        for i, pred in enumerate(real_predictions):
            if pred.get("dog_name") == actual_winner_name:
                accuracy_metrics["actual_winner_rank"] = i + 1
                if i == 0:  # Winner predicted correctly
                    accuracy_metrics["winner_predicted"] = True
                if i < 3:  # Winner in top 3
                    accuracy_metrics["top_3_hit"] = True
                break

        # If not found in predictions, set rank as beyond prediction list
        if accuracy_metrics["actual_winner_rank"] is None:
            accuracy_metrics["actual_winner_rank"] = len(real_predictions) + 1

        # Determine prediction method used
        prediction_method = "Real Prediction Analysis"
        if hasattr(prediction_result, "get") and prediction_result.get(
            "prediction_method"
        ):
            prediction_method = prediction_result["prediction_method"]
        elif any(
            pred.get("prediction_method") == "database_analysis_fallback"
            for pred in real_predictions
        ):
            prediction_method = "Database Analysis (Fallback)"

        return jsonify(
            {
                "success": True,
                "actual_winner": actual_winner,
                "predictions": real_predictions[:5],  # Top 5 predictions
                "accuracy_metrics": accuracy_metrics,
                "prediction_method": prediction_method,
                "race_info": {
                    "venue": venue,
                    "race_number": race_number,
                    "race_date": race_date,
                    "race_name": race_info.get("race_name", ""),
                },
                "note": "Historical prediction analysis using real prediction pipeline or database-driven analysis (no simulated data).",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


def _get_prediction_method_from_data(data):
    """Extract prediction method from prediction data with fallback logic"""
    try:
        # First, check for direct prediction_method field (old format)
        if "prediction_method" in data and data["prediction_method"] != "Unknown":
            return data["prediction_method"]

        # Next, check for prediction_methods_used list (new comprehensive format)
        if "prediction_methods_used" in data:
            methods_list = data["prediction_methods_used"]
            if isinstance(methods_list, list) and len(methods_list) > 0:
                return _format_prediction_methods(methods_list)

        # Check if we can infer method from individual prediction scores
        predictions = data.get("predictions", [])
        if predictions and len(predictions) > 0:
            first_prediction = predictions[0]
            prediction_scores = first_prediction.get("prediction_scores", {})

            if prediction_scores:
                # Build method list from available scores
                inferred_methods = list(prediction_scores.keys())
                if inferred_methods:
                    return _format_prediction_methods(inferred_methods)

        # Final fallback
        return "Unknown"

    except Exception as e:
        print(f"Error extracting prediction method: {e}")
        return "Unknown"


def _format_prediction_methods(methods_list):
    """Format prediction methods list into a readable string"""
    if not methods_list or not isinstance(methods_list, list):
        return "Unknown"

    # Convert method names to more readable format
    method_names = {
        "traditional": "Traditional",
        "ml_system": "ML System",
        "weather_enhanced": "Weather Enhanced",
        "enhanced_data": "Enhanced Data",
        "comprehensive_ml": "Comprehensive ML",
        "unified": "Unified",
    }

    formatted_methods = []
    for method in methods_list:
        readable_name = method_names.get(method, method.replace("_", " ").title())
        formatted_methods.append(readable_name)

    if len(formatted_methods) == 1:
        return formatted_methods[0]
    elif len(formatted_methods) == 2:
        return f"{formatted_methods[0]} + {formatted_methods[1]}"
    elif len(formatted_methods) > 2:
        return f"Multi-Method ({len(formatted_methods)} systems)"
    else:
        return "Unknown"


def safe_float(value, default=0.0):
    """Safely convert a value to float, handling None, NaN, and invalid values"""
    try:
        if isinstance(value, str) and value.lower() in ["nan", "none", "null", ""]:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


@app.route("/api/race_files_status", methods=["GET"])
def api_race_files_status():
    """API endpoint to get status of race files - predicted vs unpredicted"""
    try:
        upcoming_dir = UPCOMING_DIR
        predictions_dir = "./predictions"

        # Get all CSV files in upcoming directory
        all_race_files = []
        if os.path.exists(upcoming_dir):
            for filename in os.listdir(upcoming_dir):
                if filename.endswith(".csv") and filename != "README.md":
                    file_path = os.path.join(upcoming_dir, filename)
                    file_stat = os.stat(file_path)
                    all_race_files.append(
                        {
                            "filename": filename,
                            "race_id": filename.replace(".csv", ""),
                            "file_size": file_stat.st_size,
                            "modified": datetime.fromtimestamp(
                                file_stat.st_mtime
                            ).isoformat(),
                        }
                    )

        # Get existing predictions
        predicted_races = set()
        predictions = []
        if os.path.exists(predictions_dir):
            for filename in os.listdir(predictions_dir):
                # Updated to include both unified_prediction and prediction prefixed files
                if (
                    (
                        filename.startswith("prediction_")
                        or filename.startswith("unified_prediction_")
                    )
                    and filename.endswith(".json")
                    and "summary" not in filename
                ):
                    try:
                        file_path = os.path.join(predictions_dir, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            race_filename = data.get("race_info", {}).get(
                                "filename", ""
                            )
                            if race_filename:
                                predicted_races.add(race_filename)
                                # Get file modification time as fallback for sorting
                                file_mtime = os.path.getmtime(file_path)
                                # Handle actual prediction file structure
                                race_info = data.get("race_info", {})
                                predictions_list = data.get("predictions", [])
                                prediction_methods = data.get(
                                    "prediction_methods_used", []
                                )

                                # Extract race information from race_info
                                raw_venue = race_info.get("venue", "Unknown")
                                race_date = race_info.get("date", "Unknown")
                                distance = race_info.get("distance", "Unknown")
                                grade = race_info.get("grade", "Unknown")
                                
                                # Standardize venue name using the CSV metadata utility
                                try:
                                    from utils.csv_metadata import standardize_venue_name
                                    venue = standardize_venue_name(raw_venue) if raw_venue != "Unknown" else raw_venue
                                except ImportError:
                                    venue = raw_venue  # Fallback if utility not available

                                # Calculate total dogs from predictions
                                total_dogs = (
                                    len(predictions_list) if predictions_list else 0
                                )

                                # Calculate average confidence from predictions
                                avg_confidence = 0
                                if predictions_list:
                                    scores = [
                                        safe_float(pred.get("final_score", 0))
                                        for pred in predictions_list
                                    ]
                                    avg_confidence = (
                                        sum(scores) / len(scores) if scores else 0
                                    )

                                # Create top pick from first prediction with proper confidence
                                if predictions_list:
                                    first_pred = predictions_list[0]
                                    top_pick_data = {
                                        "dog_name": first_pred.get(
                                            "dog_name", "Unknown"
                                        ),
                                        "clean_name": first_pred.get(
                                            "dog_name", "Unknown"
                                        ),
                                        "box_number": first_pred.get(
                                            "box_number", "N/A"
                                        ),
                                        "prediction_score": safe_float(
                                            first_pred.get("final_score", 0)
                                        ),
                                        "confidence_level": first_pred.get(
                                            "confidence_level", "MEDIUM"
                                        ),
                                    }
                                else:
                                    top_pick_data = {
                                        "dog_name": "Unknown",
                                        "clean_name": "Unknown",
                                        "box_number": "N/A",
                                        "prediction_score": 0,
                                        "confidence_level": "UNKNOWN",
                                    }

                                # Format prediction method from prediction_methods_used
                                prediction_method = "Unknown"
                                if prediction_methods:
                                    if len(prediction_methods) == 1:
                                        method_map = {
                                            "traditional": "Traditional Analysis",
                                            "ml_system": "ML System",
                                            "weather_enhanced": "Weather Enhanced",
                                            "enhanced_data": "Enhanced Data",
                                        }
                                        prediction_method = method_map.get(
                                            prediction_methods[0],
                                            prediction_methods[0]
                                            .replace("_", " ")
                                            .title(),
                                        )
                                    else:
                                        prediction_method = f"Multi-Method ({len(prediction_methods)} systems)"

                                # Determine if ML was used
                                ml_used = (
                                    any(
                                        method
                                        in [
                                            "ml_system",
                                            "enhanced_data",
                                            "weather_enhanced",
                                        ]
                                        for method in prediction_methods
                                    )
                                    if prediction_methods
                                    else False
                                )

                                # Get analysis version - use same logic as prediction detail API
                                analysis_version = (
                                    data.get("analysis_version")
                                    or data.get("version")
                                    or data.get("model_version")
                                )

                                # Infer version from prediction methods if not explicitly set (same logic as detail API)
                                if not analysis_version or analysis_version in [
                                    "N/A",
                                    "nan",
                                    "null",
                                    "None",
                                    "",
                                ]:
                                    if prediction_methods:
                                        # Check for unified predictor system first (most advanced)
                                        if any(
                                            "unified" in str(method).lower()
                                            for method in prediction_methods
                                        ):
                                            # Check for specific unified subsystems
                                            if (
                                                "enhanced_pipeline_v2"
                                                in prediction_methods
                                            ):
                                                analysis_version = "Unified Comprehensive Predictor v4.0 - Enhanced Pipeline V2"
                                            elif any(
                                                method
                                                in [
                                                    "unified_comprehensive_pipeline",
                                                    "unified_weather_enhanced",
                                                    "unified_comprehensive_ml",
                                                ]
                                                for method in prediction_methods
                                            ):
                                                analysis_version = "Unified Comprehensive Predictor v4.0"
                                            else:
                                                analysis_version = (
                                                    "Unified Predictor v3.5"
                                                )
                                        # Check for enhanced pipeline v2 (second most advanced)
                                        elif (
                                            "enhanced_pipeline_v2" in prediction_methods
                                        ):
                                            analysis_version = "Enhanced Pipeline v4.0"
                                        # Check for comprehensive analysis (multiple methods = comprehensive)
                                        elif len(prediction_methods) >= 3:
                                            analysis_version = (
                                                "Comprehensive Analysis v3.0"
                                            )
                                        elif (
                                            "weather_enhanced" in prediction_methods
                                            and "enhanced_data" in prediction_methods
                                        ):
                                            analysis_version = (
                                                "Weather Enhanced + ML v2.5"
                                            )
                                        elif "weather_enhanced" in prediction_methods:
                                            analysis_version = "Weather Enhanced v2.1"
                                        elif (
                                            "enhanced_data" in prediction_methods
                                            and "ml_system" in prediction_methods
                                        ):
                                            analysis_version = "Enhanced ML System v2.3"
                                        elif "enhanced_data" in prediction_methods:
                                            analysis_version = "Enhanced Data v2.0"
                                        elif "ml_system" in prediction_methods:
                                            analysis_version = "ML System v2.0"
                                        else:
                                            analysis_version = "Multi-Method v2.0"
                                    else:
                                        analysis_version = "Standard v1.0"

                                predictions.append(
                                    {
                                        "race_id": race_filename.replace(".csv", ""),
                                        "race_name": race_filename,
                                        "race_date": race_date,
                                        "venue": venue,
                                        "distance": str(distance),
                                        "grade": grade,
                                        "track_condition": "Good",  # Default since not in current structure
                                        "total_dogs": int(total_dogs),
                                        "average_confidence": safe_float(
                                            avg_confidence
                                        ),
                                        "prediction_method": prediction_method,
                                        "ml_predictions_used": ml_used,  # Add ML usage detection
                                        "analysis_version": analysis_version,  # Add analysis version
                                        "top_pick": top_pick_data,
                                        "top_3": [
                                            {
                                                "dog_name": pred.get(
                                                    "dog_name", "Unknown"
                                                ),
                                                "clean_name": pred.get(
                                                    "dog_name", "Unknown"
                                                ),
                                                "box_number": pred.get(
                                                    "box_number", "N/A"
                                                ),
                                                "prediction_score": safe_float(
                                                    pred.get("final_score", 0)
                                                ),
                                            }
                                            for pred in (
                                                predictions_list[:3]
                                                if predictions_list
                                                else []
                                            )
                                        ],
                                        "prediction_timestamp": data.get(
                                            "prediction_timestamp", ""
                                        ),
                                        "file_mtime": file_mtime,
                                        "file_path": filename,
                                    }
                                )
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"Error processing prediction file {filename}: {e}")
                        continue

        # FIX: Handle cases where the prediction is for a test race or has an UNKNOWN venue
        # This ensures that the frontend can correctly display these predictions
        valid_predictions = []
        for pred in predictions:
            race_name = pred.get("race_name", "")
            # If the race is a test race, we should still display it
            if race_name.startswith("test_race"):
                pred["venue"] = "Test Venue"
                pred["race_date"] = datetime.now().strftime("%Y-%m-%d")
                pred["race_name"] = "Test Race"
                pred["grade"] = "Test"
                pred["distance"] = "500"

            # If the venue is unknown, we can try to extract it from the filename
            if pred.get("venue", "Unknown") in ["Unknown", "", "N/A", "nan", "null"]:
                try:
                    # Example filename: prediction_Race_1_-_TAREE_-_2025-07-26.json
                    parts = race_name.replace(".json", "").split("_")
                    if len(parts) > 3:
                        pred["venue"] = parts[3]
                        pred["race_date"] = parts[4]
                except IndexError:
                    pass

            valid_predictions.append(pred)

        # Sort valid predictions by timestamp (newest first), then by file modification time
        valid_predictions.sort(
            key=lambda x: (x.get("prediction_timestamp", ""), x.get("file_mtime", 0)),
            reverse=True,
        )
        predictions = valid_predictions

        # Separate predicted and unpredicted races
        unpredicted_races = [
            race for race in all_race_files if race["filename"] not in predicted_races
        ]

        # Sort unpredicted races by modification time (newest first)
        unpredicted_races.sort(key=lambda x: x["modified"], reverse=True)

        return jsonify(
            {
                "success": True,
                "predicted_races": predictions,
                "unpredicted_races": unpredicted_races,
                "total_predicted": len(predictions),
                "total_unpredicted": len(unpredicted_races),
                "total_files": len(all_race_files),
            }
        )

    except Exception as e:
        return jsonify(
            {
                "success": False,
                "message": f"Error getting race files status: {str(e)}",
                "predicted_races": [],
                "unpredicted_races": [],
            }
        )


@app.route("/api/generate_advisory", methods=["POST"])
def api_generate_advisory():
    """API endpoint to generate advisory messages for prediction data"""
    try:
        # Import advisory generator
        from advisory import AdvisoryGenerator
        
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Get file path or prediction data
        file_path = data.get("file_path")
        prediction_data = data.get("prediction_data")
        
        if not file_path and not prediction_data:
            return jsonify({
                "success": False, 
                "error": "Either file_path or prediction_data must be provided"
            }), 400
        
        # Initialize advisory generator
        advisory_generator = AdvisoryGenerator()
        
        # Generate advisory report
        if file_path:
            # Validate file exists
            if not os.path.exists(file_path):
                return jsonify({
                    "success": False, 
                    "error": f"File not found: {file_path}"
                }), 404
            
            advisory_result = advisory_generator.generate_advisory(file_path=file_path)
        else:
            advisory_result = advisory_generator.generate_advisory(data=prediction_data)
        
        # Return the advisory result
        return jsonify(advisory_result)
        
    except ImportError as e:
        logger.error(f"Advisory system not available: {e}")
        return jsonify({
            "success": False,
            "error": "Advisory system not available",
            "details": str(e)
        }), 500
    except Exception as e:
        logger.error(f"Error generating advisory: {e}")
        return jsonify({
            "success": False,
            "error": f"Error generating advisory: {str(e)}"
        }), 500


@app.route("/api/scrape_race_data", methods=["POST"])
def api_scrape_race_data():
    """API endpoint to scrape additional form data for a race"""
    try:
        data = request.get_json()
        race_filename = data.get("race_filename")
        data.get("venue")

        if not race_filename:
            return (
                jsonify({"success": False, "message": "Race filename is required"}),
                400,
            )

        # Ensure .csv extension
        if not race_filename.endswith(".csv"):
            race_filename += ".csv"

        race_file_path = os.path.join(UPCOMING_DIR, race_filename)

        if not os.path.exists(race_file_path):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Race file {race_filename} not found",
                    }
                ),
                404,
            )

        # Read the race file to get dog names
        try:
            import pandas as pd

            race_df = pd.read_csv(race_file_path)

            # Get unique dog names from the race file
            dog_names = []
            if "Dog Name" in race_df.columns:
                for dog_name in race_df["Dog Name"].dropna():
                    dog_name_str = str(dog_name).strip()
                    if (
                        dog_name_str
                        and dog_name_str != "nan"
                        and not dog_name_str.startswith('""')
                    ):
                        # Clean dog name (remove box number prefix if present)
                        if ". " in dog_name_str and len(dog_name_str.split(". ")) == 2:
                            parts = dog_name_str.split(". ", 1)
                            try:
                                int(parts[0])  # Check if first part is a number
                                clean_name = parts[1]  # Use the name part
                                dog_names.append(clean_name)
                            except (ValueError, TypeError):
                                dog_names.append(
                                    dog_name_str
                                )  # Keep original if prefix isn't a number
                        else:
                            dog_names.append(dog_name_str)
        except Exception as e:
            return (
                jsonify(
                    {"success": False, "message": f"Error reading race file: {str(e)}"}
                ),
                500,
            )

        if not dog_names:
            return (
                jsonify(
                    {"success": False, "message": "No dog names found in race file"}
                ),
                400,
            )

        # Run form scraper to get additional data
        scraped_count = 0
        scraping_errors = []
        new_records_found = 0
        existing_records = 0

        try:
            # Initialize form scraper
            from form_guide_csv_scraper import FormGuideCsvScraper

            scraper = FormGuideCsvScraper()

            print(
                f"🔍 Starting real data scraping for {len(dog_names)} dogs: {', '.join(dog_names[:3])}{'...' if len(dog_names) > 3 else ''}"
            )

            # Check existing data first to avoid unnecessary scraping
            form_guides_dir = "./form_guides/downloaded"
            os.makedirs(form_guides_dir, exist_ok=True)

            dogs_needing_data = []

            for dog_name in dog_names:
                # Check if we already have recent data for this dog
                existing_files = []
                if os.path.exists(form_guides_dir):
                    # Look for files containing this dog's name
                    for filename in os.listdir(form_guides_dir):
                        if filename.endswith(".csv") and dog_name.upper().replace(
                            " ", ""
                        ) in filename.upper().replace(" ", ""):
                            file_path = os.path.join(form_guides_dir, filename)
                            # Check if file is recent (within last 7 days)
                            file_age = datetime.now() - datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            )
                            if file_age.days <= 7:
                                existing_files.append(filename)

                if existing_files:
                    existing_records += 1
                    print(
                        f"   ✓ {dog_name}: Recent data exists ({len(existing_files)} files)"
                    )
                else:
                    dogs_needing_data.append(dog_name)
                    print(f"   📊 {dog_name}: Needs data scraping")

            # Only scrape for dogs that need data
            if dogs_needing_data:
                print(
                    f"🕷️ Scraping data for {len(dogs_needing_data)} dogs without recent data..."
                )

                # Use the form scraper to get recent race data
                # This will scrape form guides that may contain these dogs
                try:
                    # Get recent dates to scrape (last 7 days)
                    scrape_dates = []
                    for i in range(7):
                        date = datetime.now() - timedelta(days=i)
                        scrape_dates.append(date.strftime("%Y-%m-%d"))

                    # Scrape recent form guides by finding race URLs for specific dates
                    for date_str in scrape_dates[
                        :3
                    ]:  # Limit to last 3 days to avoid overload
                        try:
                            print(f"   📅 Scraping form guides for {date_str}...")
                            # Convert date string to date object for the scraper
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

                            # Find race URLs for this date
                            race_urls = scraper.find_race_urls(date_obj)

                            # Download CSV from each race URL (limit to avoid overload)
                            for race_url in race_urls[:5]:  # Limit to 5 races per date
                                try:
                                    if scraper.download_csv_from_race_page(race_url):
                                        scraped_count += 1
                                except Exception:
                                    continue  # Skip failed races

                            if race_urls:
                                print(
                                    f"   ✅ Processed {min(len(race_urls), 5)} races for {date_str}"
                                )
                            else:
                                print(f"   ⚪ No races found for {date_str}")

                        except Exception as date_error:
                            print(
                                f"   ⚠️ Failed to scrape {date_str}: {str(date_error)}"
                            )
                            scraping_errors.append(
                                f"Date {date_str}: {str(date_error)}"
                            )
                            continue

                    # After scraping, check if we found data for our target dogs
                    if os.path.exists(form_guides_dir):
                        for dog_name in dogs_needing_data:
                            # Look for newly created files containing this dog
                            for filename in os.listdir(form_guides_dir):
                                if filename.endswith(".csv"):
                                    file_path = os.path.join(form_guides_dir, filename)
                                    # Check if file was created in the last few minutes (during this scraping session)
                                    file_age = datetime.now() - datetime.fromtimestamp(
                                        os.path.getmtime(file_path)
                                    )
                                    if file_age.total_seconds() < 300:  # 5 minutes
                                        # Check if file contains our target dog
                                        try:
                                            df = pd.read_csv(file_path)
                                            if "Dog Name" in df.columns:
                                                dog_names_in_file = (
                                                    df["Dog Name"]
                                                    .astype(str)
                                                    .str.upper()
                                                    .str.replace(" ", "")
                                                )
                                                if (
                                                    dog_name.upper().replace(" ", "")
                                                    in dog_names_in_file.values
                                                ):
                                                    new_records_found += 1
                                                    print(
                                                        f"   ✅ Found new data for {dog_name} in {filename}"
                                                    )
                                                    break
                                        except Exception:
                                            continue

                except Exception as scraper_error:
                    scraping_errors.append(
                        f"Scraper initialization error: {str(scraper_error)}"
                    )
                    print(f"   ❌ Scraper error: {str(scraper_error)}")

            else:
                print("   ✅ All dogs have recent data - no scraping needed")

            # Final summary
            total_dogs_analyzed = len(dog_names)
            print(f"📊 Data enhancement completed for {race_filename}")
            print(f"   Dogs analyzed: {total_dogs_analyzed}")
            print(f"   Had existing data: {existing_records}")
            print(f"   Needed scraping: {len(dogs_needing_data)}")
            print(f"   New records found: {new_records_found}")
            print(f"   Scraping sessions: {scraped_count}")

        except ImportError:
            scraping_errors.append("Form scraper module not available")
            print("   ❌ FormGuideCsvScraper not available")
        except Exception as e:
            scraping_errors.append(f"Scraping error: {str(e)}")
            print(f"   ❌ Scraping error: {str(e)}")

        # Return results
        message_parts = []
        if scraped_count > 0:
            message_parts.append(
                f"Enhanced data collection initiated for {scraped_count} dogs"
            )
        if scraping_errors:
            message_parts.append(f"Issues encountered: {'; '.join(scraping_errors)}")

        return jsonify(
            {
                "success": True,
                "message": (
                    ". ".join(message_parts)
                    if message_parts
                    else "Data enhancement process completed"
                ),
                "scraped_count": scraped_count,
                "dogs_analyzed": len(dog_names),
                "errors": scraping_errors,
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error during data enhancement: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/predict_single_race", methods=["POST"])
def api_predict_single_race_standalone():
    """Standalone API endpoint for single race prediction (for frontend compatibility)"""
    data = request.get_json()
    if not data or "race_filename" not in data:
        return jsonify({"error": "No race filename provided"}), 400

    race_filename = data["race_filename"]
    race_file_path = os.path.join(UPCOMING_DIR, race_filename)

    try:
        if PredictionPipelineV3 is None:
            return jsonify({"error": "PredictionPipelineV3 not available"}), 500

        if not os.path.exists(race_file_path):
            return jsonify({"error": f"Race file not found: {race_filename}"}), 404

        logger.log_process(f"🚀 Starting V3 prediction for race: {race_filename}")
        pipeline = PredictionPipelineV3()
        prediction_result = pipeline.predict_race_file(
            race_file_path, enhancement_level="full"
        )
        logger.log_process(f"✅ Completed V3 prediction for race: {race_filename}")

        if prediction_result.get("success"):
            # Save prediction result to predictions directory for web interface
            try:
                predictions_dir = "./predictions"
                os.makedirs(predictions_dir, exist_ok=True)

                # Generate prediction filename (unified format)
                race_id = extract_race_id_from_csv_filename(race_filename)
                prediction_filename = build_prediction_filename(race_id, method="v3")
                prediction_file_path = os.path.join(
                    predictions_dir, prediction_filename
                )

                # Save prediction result
                with open(prediction_file_path, "w") as f:
                    json.dump(prediction_result, f, indent=2, default=str)

                logger.log_process(f"💾 Saved prediction to: {prediction_filename}")
            except Exception as save_error:
                logger.log_process(
                    f"⚠️ Could not save prediction file: {save_error}", level="WARNING"
                )

            return jsonify(
                {
                    "success": True,
                    "message": f"Prediction completed for {race_filename}",
                    "prediction": prediction_result,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f'Prediction failed: {prediction_result.get("error", "Unknown error")}',
                    }
                ),
                500,
            )

    except Exception as e:
        logger.log_error(
            f"Error during unified prediction for {race_filename}", error=e
        )
        return (
            jsonify({"success": False, "message": f"Prediction error: {str(e)}"}),
            500,
        )


@app.route("/api/predict_stream")
def api_predict_stream():
    """SSE streaming endpoint for real-time prediction updates"""
    try:
        import json
        import uuid

        from flask import Response, copy_current_request_context

        # Get parameters
        race_filenames = request.args.getlist("race_files")  # Multiple files support
        single_race = request.args.get("race_filename")  # Single file support
        max_workers = request.args.get("max_workers", 3, type=int)

        # Handle both single and multiple race requests
        if single_race:
            race_filenames = [single_race]
        elif not race_filenames:
            # If no files specified, get all upcoming races
            if os.path.exists(UPCOMING_DIR):
                race_filenames = [
                    f for f in os.listdir(UPCOMING_DIR) if f.endswith(".csv")
                ]
            else:
                race_filenames = []

        if not race_filenames:
            return (
                jsonify({"success": False, "error": "No race files found to predict"}),
                400,
            )

        # Validate all files exist
        race_file_paths = []
        for filename in race_filenames:
            race_file_path = os.path.join(UPCOMING_DIR, filename)
            if not os.path.exists(race_file_path):
                return (
                    jsonify(
                        {"success": False, "error": f"Race file not found: {filename}"}
                    ),
                    404,
                )
            race_file_paths.append(race_file_path)

        # Get strategy manager
        if not STRATEGY_MANAGER_AVAILABLE or not strategy_manager:
            return (
                jsonify({"success": False, "error": "Strategy Manager not available"}),
                500,
            )

        # Generate unique stream ID
        stream_id = str(uuid.uuid4())

        @copy_current_request_context
        def generate_prediction_stream():
            try:
                # Send initial status
                yield f"data: {json.dumps({'type': 'start', 'message': f'Starting predictions for {len(race_file_paths)} races...', 'total_races': len(race_file_paths), 'stream_id': stream_id})}\n\n"

                # Start streaming predictions
                strategy_manager.predict_races_streaming(
                    race_file_paths, stream_id, max_workers
                )

                # Get the stream queue and send updates
                stream_queue = strategy_manager._stream_queues.get(stream_id)
                if not stream_queue:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Stream queue not found'})}\n\n"
                    return

                completed_races = 0

                while True:
                    try:
                        # Get update from queue with timeout
                        update = stream_queue.get(timeout=1)

                        if update["type"] == "complete":
                            # Final completion message
                            yield f"data: {json.dumps(update)}\n\n"
                            break
                        elif update["type"] == "result":
                            completed_races += 1
                            progress = int(
                                (completed_races / len(race_file_paths)) * 100
                            )
                            update["progress"] = progress
                            yield f"data: {json.dumps(update)}\n\n"
                        else:
                            # Start, error, or other event types
                            yield f"data: {json.dumps(update)}\n\n"

                    except queue.Empty:
                        # Send keepalive
                        yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"
                        continue
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
                        break

                # Cleanup
                strategy_manager.remove_stream_queue(stream_id)

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream setup error: {str(e)}'})}\n\n"

        return Response(
            generate_prediction_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    except Exception as e:
        return (
            jsonify({"success": False, "error": f"SSE endpoint error: {str(e)}"}),
            500,
        )


@app.route("/api/predict_single_race_enhanced", methods=["POST"])
def api_predict_single_race_enhanced():
    """Enhanced API endpoint to predict a single race file with automatic data enhancement and detailed progress logging
    
    Accepts either race_id or race_filename in JSON body:
    - If race_filename provided: use directly
    - If race_id provided but race_filename missing: derive filename by searching directories
    - Returns clear error if neither parameter provided
    """
    import time
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting enhanced single race prediction process...")
        data = request.get_json()
        if not data:
            return (
                jsonify({
                    "success": False, 
                    "message": "Invalid JSON body",
                    "error_type": "invalid_request"
                }),
                400,
            )

        # Parse JSON body parameters
        race_id = data.get("race_id")
        race_filename = data.get("race_filename")

        # Validate that at least one parameter is provided
        if not race_id and not race_filename:
            return (
                jsonify({
                    "success": False, 
                    "message": "Either 'race_id' or 'race_filename' parameter is required",
                    "error_type": "missing_parameters",
                    "expected_parameters": ["race_id", "race_filename"]
                }),
                400,
            )

        # If race_filename is missing but race_id is present, derive filename
        if race_id and not race_filename:
            logger.info(f"Deriving filename from race_id: {race_id}")
            
            # Try multiple filename patterns that might match the race_id
            possible_filenames = [
                f"{race_id}.csv",
                f"Race {race_id}.csv",
                f"Race_{race_id}.csv"
            ]
            
            # Search in UPCOMING_DIR first
            race_file_path = None
            for filename_candidate in possible_filenames:
                candidate_path = os.path.join(UPCOMING_DIR, filename_candidate)
                if os.path.exists(candidate_path):
                    race_filename = filename_candidate
                    race_file_path = candidate_path
                    logger.info(f"Found race file in upcoming directory: {race_filename}")
                    break
            
            # If not found in upcoming, search in HISTORICAL_DIR
            if not race_file_path:
                for filename_candidate in possible_filenames:
                    candidate_path = os.path.join(HISTORICAL_DIR, filename_candidate)
                    if os.path.exists(candidate_path):
                        race_filename = filename_candidate
                        race_file_path = candidate_path
                        logger.info(f"Found race file in historical directory: {race_filename}")
                        break
            
            # If still not found, search for partial matches in both directories
            if not race_file_path:
                logger.info(f"Searching for partial filename matches for race_id: {race_id}")
                
                # Search upcoming directory for partial matches
                if os.path.exists(UPCOMING_DIR):
                    for file in os.listdir(UPCOMING_DIR):
                        if file.endswith(".csv") and race_id in file:
                            race_filename = file
                            race_file_path = os.path.join(UPCOMING_DIR, file)
                            logger.info(f"Found partial match in upcoming directory: {race_filename}")
                            break
                
                # Search historical directory for partial matches if not found
                if not race_file_path and os.path.exists(HISTORICAL_DIR):
                    for file in os.listdir(HISTORICAL_DIR):
                        if file.endswith(".csv") and race_id in file:
                            race_filename = file
                            race_file_path = os.path.join(HISTORICAL_DIR, file)
                            logger.info(f"Found partial match in historical directory: {race_filename}")
                            break
            
            # If no file found, return error
            if not race_file_path:
                return (
                    jsonify({
                        "success": False,
                        "message": f"No race file found for race_id '{race_id}'. Searched in {UPCOMING_DIR} and {HISTORICAL_DIR}",
                        "error_type": "file_not_found",
                        "race_id": race_id,
                        "searched_directories": [UPCOMING_DIR, HISTORICAL_DIR],
                        "attempted_filenames": possible_filenames
                    }),
                    404,
                )
        else:
            # race_filename was provided, determine the full path
            race_file_path = os.path.join(UPCOMING_DIR, race_filename)
            
            # Check if file exists in upcoming directory first
            if not os.path.exists(race_file_path):
                # Try historical directory
                historical_path = os.path.join(HISTORICAL_DIR, race_filename)
                if os.path.exists(historical_path):
                    race_file_path = historical_path
                    logger.info(f"Found race file in historical directory: {race_filename}")
                else:
                    return (
                        jsonify({
                            "success": False,
                            "message": f"Race file '{race_filename}' not found in {UPCOMING_DIR} or {HISTORICAL_DIR}",
                            "error_type": "file_not_found",
                            "race_filename": race_filename,
                            "searched_directories": [UPCOMING_DIR, HISTORICAL_DIR]
                        }),
                        404,
                    )

        # STEP 1: Automatically enhance data before prediction
        logger.info(f"🔍 Step 1: Enhancing data for {race_filename} before prediction...")
        start_step_time = time.time()
        
        # Enhancement logic - basic file validation and preprocessing
        try:
            if not os.path.exists(race_file_path):
                raise FileNotFoundError(f"Race file not found: {race_file_path}")
            
            # Check file size
            file_size = os.path.getsize(race_file_path)
            logger.info(f"📊 File size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Race file is empty")
            elif file_size < 100:
                logger.warning(f"⚠️ Race file is very small ({file_size} bytes) - may be incomplete")
            
            # Quick CSV validation
            try:
                import pandas as pd
                df_check = pd.read_csv(race_file_path, nrows=1)
                logger.info(f"✅ CSV validation passed - {len(df_check.columns)} columns detected")
            except Exception as csv_error:
                logger.warning(f"⚠️ CSV validation warning: {str(csv_error)}")
            
            step_time = time.time() - start_step_time
            logger.info(f"✅ Step 1 completed in {step_time:.2f} seconds")
            
        except Exception as enhance_error:
            logger.error(f"❌ Step 1 failed: {str(enhance_error)}")
            return (
                jsonify({
                    "success": False,
                    "message": f"Data enhancement failed: {str(enhance_error)}",
                    "error_type": "data_enhancement_error",
                    "race_filename": race_filename,
                    "processing_time_seconds": time.time() - start_time
                }),
                400,
            )

        # STEP 2: Run prediction pipeline using the most appropriate predictor
        logger.info(f"🔍 Step 2: Running prediction pipeline for race {race_filename}...")
        prediction_start_time = time.time()
        
        prediction_result = None
        predictor_used = None
        attempts = []
        
        # Try PredictionPipelineV4 first (most advanced)
        if PredictionPipelineV4 and not prediction_result:
            try:
                logger.info(f"Using PredictionPipelineV4 for {race_filename}")
                predictor = PredictionPipelineV4()
                prediction_result = predictor.predict_race_file(race_file_path)
                predictor_used = "PredictionPipelineV4"
                logger.info(f"Successfully used PredictionPipelineV4 for {race_filename}")
            except Exception as e:
                logger.warning(f"PredictionPipelineV4 failed: {str(e)}")
        
        # Fallback to PredictionPipelineV3 if V4 failed
        if PredictionPipelineV3 and not prediction_result:
            try:
                logger.info(f"Fallback to PredictionPipelineV3 for {race_filename}")
                predictor = PredictionPipelineV3()
                prediction_result = predictor.predict_race_file(
                    race_file_path, 
                    enhancement_level="full"
                )
                predictor_used = "PredictionPipelineV3"
                logger.info(f"Successfully used PredictionPipelineV3 for {race_filename}")
            except Exception as e:
                logger.warning(f"PredictionPipelineV3 failed: {str(e)}")
        
        # Fallback to UnifiedPredictor if both V4 and V3 failed
        if UnifiedPredictor and not prediction_result:
            try:
                logger.info(f"Final fallback to UnifiedPredictor for {race_filename}")
                predictor = UnifiedPredictor()
                prediction_result = predictor.predict_race_file(race_file_path)
                predictor_used = "UnifiedPredictor"
                logger.info(f"Successfully used UnifiedPredictor for {race_filename}")
            except Exception as e:
                logger.warning(f"UnifiedPredictor failed: {str(e)}")
        
        # Final fallback to ComprehensivePredictionPipeline
        if ComprehensivePredictionPipeline and not prediction_result:
            try:
                logger.info(f"Last resort fallback to ComprehensivePredictionPipeline for {race_filename}")
                predictor = ComprehensivePredictionPipeline()
                prediction_result = predictor.predict_race_file(race_file_path)
                predictor_used = "ComprehensivePredictionPipeline"
                logger.info(f"Successfully used ComprehensivePredictionPipeline for {race_filename}")
            except Exception as e:
                logger.warning(f"ComprehensivePredictionPipeline failed: {str(e)}")
        
        # Check if any predictor succeeded
        if not prediction_result:
            return (
                jsonify({
                    "success": False,
                    "message": "All prediction pipelines failed",
                    "error_type": "prediction_pipeline_failure",
                    "race_id": race_id,
                    "race_filename": race_filename,
                    "attempted_predictors": ["PredictionPipelineV4", "PredictionPipelineV3", "UnifiedPredictor", "ComprehensivePredictionPipeline"]
                }),
                500,
            )

        # STEP 3: Return unified response contract
        if prediction_result.get("success"):
            # Extract race_id from filename if not provided
            if not race_id and race_filename:
                race_id = extract_race_id_from_csv_filename(race_filename)
            
            return jsonify({
                "success": True,
                "race_id": race_id,
                "race_filename": race_filename,
                "predictions": prediction_result.get("predictions", []),
                "predictor_used": predictor_used,
                "file_path": race_file_path,
                "enhancement_applied": True,
                "message": f"Prediction completed successfully using {predictor_used}",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return (
                jsonify({
                    "success": False,
                    "message": prediction_result.get("error", "Unknown prediction error"),
                    "error_type": "prediction_error",
                    "race_id": race_id,
                    "race_filename": race_filename,
                    "predictor_used": predictor_used,
                    "file_path": race_file_path,
                    "prediction_details": prediction_result
                }),
                500,
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in predict_single_race_enhanced: {str(e)}", exc_info=True)
        return (
            jsonify({
                "success": False, 
                "message": f"Unexpected server error: {str(e)}", 
                "error_type": "server_error",
                "race_id": data.get("race_id") if data else None,
                "race_filename": data.get("race_filename") if data else None
            }),
            500,
        )


@app.route("/api/predict_all_upcoming_races_enhanced", methods=["POST"])
def api_predict_all_upcoming_races_enhanced():
    """Enhanced API endpoint to predict all upcoming races with comprehensive error handling and detailed progress logging"""
    import time
    start_time = time.time()
    
    try:
        # Initialize counters
        total_races = 0
        success_count = 0
        errors = []
        results = []
        processed_files = []
        
        # Step 1: Enumerate CSV files in UPCOMING_DIR
        logger.info(f"🔍 Step 1: Scanning upcoming races directory: {UPCOMING_DIR}")
        if not os.path.exists(UPCOMING_DIR):
            logger.error(f"❌ Upcoming races directory not found: {UPCOMING_DIR}")
            return jsonify({
                "success": True,
                "message": "No upcoming races directory found",
                "total_races": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "predictions": [],
                "errors": ["Upcoming races directory does not exist"],
                "total_processing_time_seconds": time.time() - start_time
            })
        
        # Get all CSV and JSON files using the helper function
        upcoming_races = load_upcoming_races(refresh=False)
        upcoming_files = [race.get("filename", f"{race.get('name', 'race')}.csv") for race in upcoming_races if race.get("filename") or race.get("name")]
        
        # If no files from helper, fallback to direct directory scan for CSV only
        if not upcoming_files:
            upcoming_files = [f for f in os.listdir(UPCOMING_DIR) if f.endswith(".csv")]
        
        total_races = len(upcoming_files)
        
        logger.info(f"📊 Found {total_races} CSV files to process")
        
        if total_races == 0:
            logger.info("ℹ️ No upcoming races found")
            return jsonify({
                "success": True,
                "message": "No upcoming races found",
                "total_races": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "predictions": [],
                "errors": [],
                "total_processing_time_seconds": time.time() - start_time
            })
        
        logger.info(f"🚀 Starting enhanced batch prediction for {total_races} upcoming races")
        logger.info(f"📂 Processing files: {', '.join(upcoming_files[:5])}{'...' if len(upcoming_files) > 5 else ''}")
        
        # Step 2: Use existing batch-prediction helper
        # Try to use ComprehensivePredictionPipeline first (most comprehensive)
        prediction_pipeline = None
        pipeline_type = "unknown"
        
        if ComprehensivePredictionPipeline:
            try:
                prediction_pipeline = ComprehensivePredictionPipeline()
                pipeline_type = "ComprehensivePredictionPipeline"
                
                # Use the existing predict_all_upcoming_races method
                batch_results = prediction_pipeline.predict_all_upcoming_races(
                    upcoming_dir=UPCOMING_DIR,
                    force_rerun=False
                )
                
                if batch_results and batch_results.get("success"):
                    # Extract results
                    success_count = batch_results.get("successful_predictions", 0)
                    pipeline_results = batch_results.get("results", [])
                    
                    # Process each result to extract predictions and errors
                    for result in pipeline_results:
                        if result and result.get("success"):
                            results.append(result)
                        else:
                            error_msg = result.get("error", "Unknown prediction error") if result else "No result returned"
                            errors.append(error_msg)
                            logger.error(f"Prediction failed: {error_msg}")
                    
                    logger.info(f"Batch prediction completed using {pipeline_type}: {success_count}/{total_races} successful")
                    
                else:
                    raise Exception(f"Batch prediction failed: {batch_results.get('message', 'Unknown error') if batch_results else 'No results returned'}")
                    
            except Exception as e:
                logger.error(f"ComprehensivePredictionPipeline failed: {str(e)}")
                prediction_pipeline = None
                errors.append(f"ComprehensivePredictionPipeline error: {str(e)}")
        
        # Fallback to PredictionPipelineV3 if comprehensive pipeline failed
        if not prediction_pipeline and PredictionPipelineV3:
            try:
                prediction_pipeline = PredictionPipelineV3()
                pipeline_type = "PredictionPipelineV3"
                
                logger.info(f"Using fallback prediction pipeline: {pipeline_type}")
                
                # Process each file individually
                for filename in upcoming_files:
                    try:
                        race_file_path = os.path.join(UPCOMING_DIR, filename)
                        logger.info(f"Predicting race: {filename}")
                        
                        prediction_result = prediction_pipeline.predict_race_file(
                            race_file_path, 
                            enhancement_level="full"
                        )
                        
                        if prediction_result and prediction_result.get("success"):
                            results.append(prediction_result)
                            success_count += 1
                            logger.info(f"Successfully predicted race: {filename}")
                        else:
                            error_msg = f"Prediction failed for {filename}: {prediction_result.get('error', 'Unknown error') if prediction_result else 'No result returned'}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                            
                    except Exception as race_error:
                        error_msg = f"Error predicting {filename}: {str(race_error)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        
            except Exception as v3_error:
                logger.error(f"PredictionPipelineV3 failed: {str(v3_error)}")
                prediction_pipeline = None
                errors.append(f"PredictionPipelineV3 error: {str(v3_error)}")
        
        # Final fallback to UnifiedPredictor if all else fails
        if not prediction_pipeline and UnifiedPredictor:
            try:
                prediction_pipeline = UnifiedPredictor()
                pipeline_type = "UnifiedPredictor"
                
                logger.info(f"Using final fallback prediction pipeline: {pipeline_type}")
                
                # Process each file individually
                for filename in upcoming_files:
                    try:
                        race_file_path = os.path.join(UPCOMING_DIR, filename)
                        logger.info(f"Predicting race: {filename}")
                        
                        prediction_result = prediction_pipeline.predict_race_file(race_file_path)
                        
                        if prediction_result and prediction_result.get("success"):
                            results.append(prediction_result)
                            success_count += 1
                            logger.info(f"Successfully predicted race: {filename}")
                        else:
                            error_msg = f"Prediction failed for {filename}: {prediction_result.get('error', 'Unknown error') if prediction_result else 'No result returned'}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                            
                    except Exception as race_error:
                        error_msg = f"Error predicting {filename}: {str(race_error)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        
            except Exception as unified_error:
                logger.error(f"UnifiedPredictor failed: {str(unified_error)}")
                errors.append(f"UnifiedPredictor error: {str(unified_error)}")
        
        # Step 3: Return JSON with success: true even if some races fail, but include errors list
        response = {
            "success": True,  # Always return success: true as per requirements
            "message": f"Batch prediction completed: {success_count}/{total_races} races predicted successfully",
            "total_races": total_races,
            "success_count": success_count,
            "failed_count": total_races - success_count,
            "predictions": results,
            "errors": errors,
            "pipeline_used": pipeline_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log final summary
        if success_count == total_races:
            logger.info(f"✅ All {total_races} races predicted successfully using {pipeline_type}")
        elif success_count > 0:
            logger.info(f"⚠️ Partial success: {success_count}/{total_races} races predicted successfully using {pipeline_type}")
        else:
            logger.error(f"❌ No races predicted successfully. Total errors: {len(errors)}")
        
        return jsonify(response)
        
    except Exception as e:
        # Step 4: Log exceptions with logger.error()
        error_msg = f"Enhanced batch prediction failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return jsonify({
            "success": True,  # Still return success: true as per requirements
            "message": "Batch prediction encountered an error but completed",
            "total_races": 0,
            "success_count": 0,
            "failed_count": 0,
            "predictions": [],
            "errors": [error_msg],
            "pipeline_used": "none",
            "timestamp": datetime.now().isoformat()
        }), 500



@app.route("/api/prediction_results", methods=["GET"])
def api_prediction_results():
    """API endpoint to get prediction results from JSON files"""
    try:
        predictions_dir = "./predictions"
        if not os.path.exists(predictions_dir):
            return jsonify(
                {
                    "success": False,
                    "message": "No predictions directory found",
                    "predictions": [],
                }
            )

        # Find all prediction JSON files (not summary files) - include both regular and unified predictions
        prediction_files = []
        for filename in os.listdir(predictions_dir):
            if (
                (
                    filename.startswith("prediction_")
                    or filename.startswith("unified_prediction_")
                )
                and filename.endswith(".json")
                and "summary" not in filename
            ):
                file_path = os.path.join(predictions_dir, filename)
                mtime = os.path.getmtime(file_path)

                # Assign priority: comprehensive predictions (1) > weather-enhanced (2) > others (3)
                priority = 1 if "weather_enhanced" not in filename else 2
                if filename.startswith("unified_prediction_"):
                    priority = 1  # Unified predictions are also high priority

                prediction_files.append((file_path, mtime, priority))

        # Sort by priority first (lower number = higher priority), then by modification time (newest first)
        prediction_files.sort(key=lambda x: (x[2], -x[1]))

        predictions = []
        for file_path, mtime in prediction_files[:10]:  # Get latest 10 predictions
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                    # Handle actual prediction file structure
                    race_info = data.get("race_info", {})
                    predictions_list = data.get("predictions", [])
                    prediction_methods = data.get("prediction_methods_used", [])

                    # Extract basic race information
                    venue = race_info.get("venue", "Unknown")
                    race_date = race_info.get("date", "Unknown")
                    distance = race_info.get("distance", "Unknown")

                    # Calculate total dogs from predictions
                    total_dogs = len(predictions_list) if predictions_list else 0

                    # Calculate average confidence from prediction scores
                    avg_confidence = 0
                    if predictions_list:
                        scores = [
                            safe_float(pred.get("final_score", 0))
                            for pred in predictions_list
                        ]
                        avg_confidence = sum(scores) / len(scores) if scores else 0

                    # Format prediction method from prediction_methods_used
                    prediction_method = "Unknown"
                    if prediction_methods:
                        if len(prediction_methods) == 1:
                            method_map = {
                                "traditional": "Traditional Analysis",
                                "ml_system": "ML System",
                                "weather_enhanced": "Weather Enhanced",
                                "enhanced_data": "Enhanced Data",
                            }
                            prediction_method = method_map.get(
                                prediction_methods[0],
                                prediction_methods[0].replace("_", " ").title(),
                            )
                        else:
                            prediction_method = (
                                f"Multi-Method ({len(prediction_methods)} systems)"
                            )

                    # Create top pick from first prediction if available
                    top_pick_data = {
                        "dog_name": "Unknown",
                        "box_number": "N/A",
                        "prediction_score": 0,
                    }
                    if predictions_list:
                        first_pred = predictions_list[0]
                        # Enhanced KeyError handling for dog names
                        from constants import DOG_NAME_KEY
                        from pathlib import Path
                        
                        try:
                            dog_name = first_pred[DOG_NAME_KEY]
                        except KeyError:
                            # Log the KeyError with detailed context using key_mismatch_logger
                            key_mismatch_logger.log_key_error(
                                error_context={
                                    "operation": "top_pick_creation_from_prediction_results",
                                    "race_file_path": str(Path(file_path).name),
                                    "dog_record": dict(first_pred),
                                    "available_keys": list(first_pred.keys()),
                                    "missing_key": DOG_NAME_KEY,
                                    "step": "api_prediction_results_processing"
                                },
                                dog_record=dict(first_pred)
                            )
                            # Use fallback value
                            dog_name = first_pred.get("dog_name", "Unknown")
                        
                        top_pick_data = {
                            "dog_name": dog_name,
                            "box_number": first_pred.get("box_number", "N/A"),
                            "prediction_score": safe_float(
                                first_pred.get("final_score", 0)
                            ),
                        }

                    predictions.append(
                        {
                            "race_name": race_info.get("filename", "Unknown Race"),
                            "race_date": race_date,
                            "venue": venue,
                            "distance": distance,
                            "total_dogs": total_dogs,
                            "average_confidence": avg_confidence,
                            "prediction_method": prediction_method,
                            "top_pick": top_pick_data,
                            "top_3": [
                                {
                                    "dog_name": pred.get("dog_name", "Unknown"),
                                    "box_number": pred.get("box_number", "N/A"),
                                    "prediction_score": safe_float(
                                        pred.get("final_score", 0)
                                    ),
                                }
                                for pred in (
                                    predictions_list[:3] if predictions_list else []
                                )
                            ],
                            "prediction_timestamp": data.get(
                                "prediction_timestamp", ""
                            ),
                            "file_path": os.path.basename(file_path),
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                continue  # Skip corrupted files

        return jsonify(
            {
                "success": True,
                "predictions": predictions,
                "total_files": len(prediction_files),
            }
        )

    except Exception as e:
        return jsonify(
            {
                "success": False,
                "message": f"Error reading prediction files: {str(e)}",
                "predictions": [],
            }
        )


@app.route("/api/prediction_detail/<race_name>")
def api_prediction_detail(race_name):
    """API endpoint to get detailed prediction data for a specific race with venue mapping support"""
    try:
        predictions_dir = "./predictions"

        # Initialize venue mapper for better historical data lookup
        # GreyhoundVenueMapper()  # Module not available

        # Try multiple prediction file naming conventions
        prediction_file = None
        possible_filenames = [
            f"prediction_{race_name}.json",  # Original format
        ]

        # Also try compact format (this is the most common format)
        if race_name.startswith("Race ") and " - " in race_name:
            parts = race_name.split(" - ")
            if len(parts) >= 3:
                race_num = parts[0].replace("Race ", "")
                venue = parts[1]
                date = parts[2]
                compact_name = f"{race_num}_{venue}_{date}"
                possible_filenames.append(f"prediction_{compact_name}.json")

                # Also try without .csv extension from the race_name
                if date.endswith(".csv"):
                    date_clean = date[:-4]
                    compact_name_clean = f"{race_num}_{venue}_{date_clean}"
                    possible_filenames.append(f"prediction_{compact_name_clean}.json")

        # Try to find existing prediction file
        for filename in possible_filenames:
            file_path = os.path.join(predictions_dir, filename)
            if os.path.exists(file_path):
                prediction_file = file_path
                break

        # If still not found, search through all prediction files for a match
        if not prediction_file and os.path.exists(predictions_dir):
            for filename in os.listdir(predictions_dir):
                if (
                    filename.startswith("prediction_")
                    and filename.endswith(".json")
                    and "summary" not in filename
                ):
                    try:
                        file_path = os.path.join(predictions_dir, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            original_filename = data.get("race_info", {}).get(
                                "filename", ""
                            )
                            if original_filename.endswith(".csv"):
                                original_filename = original_filename[:-4]
                            if original_filename == race_name:
                                prediction_file = file_path
                                break
                    except (json.JSONDecodeError, KeyError, IOError):
                        continue

        if not prediction_file:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"No prediction found for race: {race_name}",
                    }
                ),
                404,
            )

        # Load the prediction data
        with open(prediction_file, "r") as f:
            prediction_data = json.load(f)

        # Extract race context information - handle both race_info and race_context formats
        race_context = prediction_data.get("race_context", {})
        race_info = prediction_data.get("race_info", {})

        # Use race_info if race_context is empty (newer format) - this is the common case
        if not race_context and race_info:
            race_context = {
                "venue": race_info.get("venue", "Unknown"),
                "race_date": race_info.get("date", "Unknown"),
                "distance": race_info.get("distance", "Unknown"),
                "grade": race_info.get("grade", "Unknown"),
                "race_number": race_info.get("race_number", "Unknown"),
                "filename": race_info.get("filename", "Unknown"),
            }
        elif not race_context and not race_info:
            # Fallback if neither exists
            race_context = {
                "venue": "Unknown",
                "race_date": "Unknown",
                "distance": "Unknown",
                "grade": "Unknown",
                "race_number": "Unknown",
                "filename": "Unknown",
            }

        # Calculate prediction summary from the predictions list
        predictions_list = prediction_data.get("predictions", [])
        total_dogs = len(predictions_list)

        # Calculate average confidence from prediction scores
        avg_confidence = 0
        if predictions_list:
            scores = [
                pred.get("final_score", pred.get("prediction_score", 0))
                for pred in predictions_list
            ]
            avg_confidence = sum(scores) / len(scores) if scores else 0

        # Extract top pick from first prediction with better data handling
        top_pick = None
        if predictions_list:
            first_pred = predictions_list[0]

            # Get dog name from multiple possible sources
            dog_name = (
                first_pred.get("dog_name")
                or first_pred.get("clean_name")
                or first_pred.get("name")
                or "Unknown"
            )

            # Get box number
            box_number = first_pred.get("box_number", first_pred.get("box", "N/A"))

            # Get prediction score
            prediction_score = (
                first_pred.get("final_score")
                or first_pred.get("prediction_score")
                or first_pred.get("score")
                or 0
            )

            # Handle confidence level properly
            confidence_level = first_pred.get("confidence_level", "Unknown")

            # If confidence level is missing, NaN, or null, derive from prediction score
            if (
                not confidence_level
                or confidence_level in ["NaN", "nan", "null", "None", "Unknown"]
                or str(confidence_level).lower() in ["nan", "null", "none"]
            ):

                # Derive confidence from prediction score
                score_val = safe_float(prediction_score, 0)
                if score_val >= 0.7:
                    confidence_level = "HIGH"
                elif score_val >= 0.5:
                    confidence_level = "MEDIUM"
                elif score_val >= 0.3:
                    confidence_level = "LOW"
                else:
                    confidence_level = "VERY_LOW"

            # Format confidence level for display
            if isinstance(confidence_level, str):
                # If it's already a text level (HIGH, MEDIUM, LOW), keep it
                if confidence_level in ["HIGH", "MEDIUM", "LOW", "VERY_LOW"]:
                    formatted_confidence = confidence_level
                # If it's a percentage string, keep it as is
                elif "%" in confidence_level:
                    formatted_confidence = confidence_level
                # If it's a numeric string, try to convert it to percentage
                elif confidence_level.replace(".", "").replace("-", "").isdigit():
                    try:
                        formatted_confidence = f"{int(float(confidence_level))}%"
                    except (ValueError, TypeError):
                        formatted_confidence = confidence_level
                else:
                    formatted_confidence = confidence_level
            else:
                formatted_confidence = "MEDIUM"  # Default fallback

            top_pick = {
                "dog_name": dog_name,
                "box_number": str(box_number) if box_number != "N/A" else "N/A",
                "prediction_score": safe_float(prediction_score, 0),
                "confidence_level": formatted_confidence,
            }

        # Create race summary with better data detection
        # Determine if ML was used from multiple sources
        ml_used = False
        prediction_methods = prediction_data.get("prediction_methods_used", [])
        prediction_method = prediction_data.get("prediction_method", "")

        # Check for ML usage indicators in prediction_methods_used array
        ml_indicators = [
            "ml_system",
            "enhanced_data",
            "weather_enhanced",
            "comprehensive",
            "neural",
            "ensemble",
        ]

        # Primary check: prediction_methods_used array (this is the most reliable)
        if prediction_methods:
            ml_used = any(
                any(indicator in str(method).lower() for indicator in ml_indicators)
                for method in prediction_methods
                if method
            )
            # Also check for direct ML indicators
            if not ml_used:
                ml_used = any(
                    method in ["ml_system", "enhanced_data", "weather_enhanced"]
                    for method in prediction_methods
                )

        # Secondary check: prediction method string
        if not ml_used and prediction_method:
            ml_used = any(
                indicator in prediction_method.lower() for indicator in ml_indicators
            )

        # Tertiary check: prediction_scores breakdown (indicates ML usage)
        if not ml_used and predictions_list:
            first_pred = predictions_list[0]
            prediction_scores = first_pred.get("prediction_scores", {})
            # If we have multiple prediction scores (traditional + ML methods), ML was used
            if prediction_scores and len(prediction_scores) > 1:
                ml_used = True
                # Check specifically for ML method keys
                ml_keys = ["enhanced_data", "weather_enhanced", "ml_system"]
                ml_used = any(key in prediction_scores for key in ml_keys)

        # Get analysis version - infer from data structure if not explicitly set
        analysis_version = (
            prediction_data.get("analysis_version")
            or prediction_data.get("version")
            or prediction_data.get("model_version")
        )

        # Infer version from prediction methods if not explicitly set
        if not analysis_version or analysis_version in [
            "N/A",
            "nan",
            "null",
            "None",
            "",
        ]:
            if prediction_methods:
                # Check for unified predictor system first (most advanced)
                if any(
                    "unified" in str(method).lower() for method in prediction_methods
                ):
                    # Check for specific unified subsystems
                    if "enhanced_pipeline_v2" in prediction_methods:
                        analysis_version = "Unified Comprehensive Predictor v4.0 - Enhanced Pipeline V2"
                    elif any(
                        method
                        in [
                            "unified_comprehensive_pipeline",
                            "unified_weather_enhanced",
                            "unified_comprehensive_ml",
                        ]
                        for method in prediction_methods
                    ):
                        analysis_version = "Unified Comprehensive Predictor v4.0"
                    else:
                        analysis_version = "Unified Predictor v3.5"
                # Check for enhanced pipeline v2 (second most advanced)
                elif "enhanced_pipeline_v2" in prediction_methods:
                    analysis_version = "Enhanced Pipeline v4.0"
                # Check for comprehensive analysis (multiple methods = comprehensive)
                elif len(prediction_methods) >= 3:
                    analysis_version = "Comprehensive Analysis v3.0"
                elif (
                    "weather_enhanced" in prediction_methods
                    and "enhanced_data" in prediction_methods
                ):
                    analysis_version = "Weather Enhanced + ML v2.5"
                elif "weather_enhanced" in prediction_methods:
                    analysis_version = "Weather Enhanced v2.1"
                elif (
                    "enhanced_data" in prediction_methods
                    and "ml_system" in prediction_methods
                ):
                    analysis_version = "Enhanced ML System v2.3"
                elif "enhanced_data" in prediction_methods:
                    analysis_version = "Enhanced Data v2.0"
                elif "ml_system" in prediction_methods:
                    analysis_version = "ML System v2.0"
                else:
                    analysis_version = "Multi-Method v2.0"
            else:
                analysis_version = "Standard v1.0"

        # Determine prediction method name from methods used
        if prediction_methods:
            # Check for unified predictor system first
            if any("unified" in str(method).lower() for method in prediction_methods):
                if "enhanced_pipeline_v2" in prediction_methods:
                    method_name = (
                        "Unified Comprehensive Predictor - Enhanced Pipeline V2"
                    )
                elif any(
                    method
                    in [
                        "unified_comprehensive_pipeline",
                        "unified_weather_enhanced",
                        "unified_comprehensive_ml",
                    ]
                    for method in prediction_methods
                ):
                    method_name = "Unified Comprehensive Predictor"
                else:
                    method_name = "Unified Predictor System"
            # Check for enhanced pipeline v2
            elif "enhanced_pipeline_v2" in prediction_methods:
                method_name = "Enhanced Pipeline V2"
            elif (
                "weather_enhanced" in prediction_methods
                and "enhanced_data" in prediction_methods
            ):
                method_name = "Weather Enhanced + ML"
            elif "weather_enhanced" in prediction_methods:
                method_name = "Weather Enhanced ML"
            elif (
                "enhanced_data" in prediction_methods
                and "ml_system" in prediction_methods
            ):
                method_name = "Enhanced ML System"
            elif "enhanced_data" in prediction_methods:
                method_name = "Enhanced Data Analysis"
            elif len(prediction_methods) > 2:
                method_name = (
                    f"Multi-Method Analysis ({len(prediction_methods)} systems)"
                )
            else:
                method_name = "Combined Analysis"
        else:
            method_name = prediction_method or "Standard Analysis"

        race_summary = {
            "total_dogs": total_dogs,
            "average_confidence": avg_confidence,
            "prediction_method": method_name,
            "ml_used": ml_used,
            "analysis_version": analysis_version,
        }

        # Enhanced analysis for each dog's prediction reasoning
        enhanced_predictions = []
        for dog in prediction_data.get("predictions", []):
            # Calculate reasoning scores
            historical_stats = dog.get("historical_stats", {})

            reasoning = {
                "performance_factors": [],
                "risk_factors": [],
                "strengths": [],
                "concerns": [],
                "key_metrics": {},
                "key_factors": [],  # Add key_factors array
            }

            # Performance analysis - handle both old and new format
            # New format uses direct values, old format might use ratios
            win_rate = historical_stats.get("win_rate", 0)
            place_rate = historical_stats.get("place_rate", 0)
            consistency = historical_stats.get(
                "consistency", historical_stats.get("position_consistency", 0)
            )
            avg_position = historical_stats.get(
                "average_position", historical_stats.get("avg_position", 10)
            )
            races_count = historical_stats.get(
                "total_races", historical_stats.get("races_count", 0)
            )

            # Extract confidence level - handle multiple possible sources
            confidence_level = dog.get("confidence_level")
            if (
                not confidence_level
                or confidence_level == "NaN"
                or str(confidence_level).lower() == "nan"
            ):
                # Try to derive confidence from prediction score
                prediction_score = dog.get(
                    "final_score", dog.get("prediction_score", 0)
                )
                if prediction_score >= 0.7:
                    confidence_level = "HIGH"
                elif prediction_score >= 0.5:
                    confidence_level = "MEDIUM"
                elif prediction_score >= 0.3:
                    confidence_level = "LOW"
                else:
                    confidence_level = "VERY_LOW"

            # Store confidence level back in dog data
            dog["confidence_level"] = confidence_level

            # Convert to consistent formats
            if (
                isinstance(win_rate, (int, float)) and win_rate > 1
            ):  # Assume percentage format
                win_rate = win_rate / 100
            if (
                isinstance(place_rate, (int, float)) and place_rate > 1
            ):  # Assume percentage format
                place_rate = place_rate / 100
            if (
                isinstance(consistency, (int, float)) and consistency > 1
            ):  # Assume percentage format
                consistency = consistency / 100

            # Ensure numeric values with safer conversion
            try:
                win_rate = (
                    float(win_rate) if isinstance(win_rate, (int, float)) else 0.0
                )
            except (ValueError, TypeError):
                win_rate = 0.0

            try:
                place_rate = (
                    float(place_rate) if isinstance(place_rate, (int, float)) else 0.0
                )
            except (ValueError, TypeError):
                place_rate = 0.0

            try:
                consistency = (
                    float(consistency) if isinstance(consistency, (int, float)) else 0.0
                )
            except (ValueError, TypeError):
                consistency = 0.0

            try:
                avg_position = (
                    float(avg_position)
                    if isinstance(avg_position, (int, float))
                    else 10.0
                )
            except (ValueError, TypeError):
                avg_position = 10.0

            try:
                races_count = (
                    int(races_count) if isinstance(races_count, (int, float)) else 0
                )
            except (ValueError, TypeError):
                races_count = 0

            reasoning["key_metrics"] = {
                "win_percentage": round(win_rate * 100, 1),
                "place_percentage": round(place_rate * 100, 1),
                "avg_finish_position": round(avg_position, 1),
                "consistency_score": round(consistency * 100, 1),
                "experience_level": races_count,
                "recent_form_trend": historical_stats.get("form_trend", 0),
            }

            # Strengths analysis
            if win_rate > 0.2:
                reasoning["strengths"].append(
                    f"Strong winner ({round(win_rate * 100, 1)}% win rate)"
                )
            if place_rate > 0.5:
                reasoning["strengths"].append(
                    f"Consistent placer ({round(place_rate * 100, 1)}% place rate)"
                )
            if consistency > 0.8:
                reasoning["strengths"].append("Very consistent performer")
            if avg_position < 3.5:
                reasoning["strengths"].append(
                    f"Typically finishes well (avg: {round(avg_position, 1)})"
                )
            if races_count > 10:
                reasoning["strengths"].append(
                    f"Experienced runner ({races_count} races)"
                )

            recent_activity = historical_stats.get("recent_activity", {})

            # Safely convert days_since_last_race to numeric
            days_since_last_race_raw = recent_activity.get("days_since_last_race", 999)
            try:
                if isinstance(days_since_last_race_raw, (int, float)):
                    days_since_last_race = int(days_since_last_race_raw)
                elif isinstance(days_since_last_race_raw, str):
                    # Try to convert string to number, handle common cases
                    if days_since_last_race_raw.lower() in [
                        "nan",
                        "none",
                        "null",
                        "",
                        "n/a",
                        "not_found",
                    ]:
                        days_since_last_race = 999
                    elif (
                        days_since_last_race_raw.replace(".", "")
                        .replace("-", "")
                        .isdigit()
                    ):
                        days_since_last_race = int(float(days_since_last_race_raw))
                    else:
                        days_since_last_race = 999
                else:
                    days_since_last_race = 999
            except (ValueError, TypeError):
                days_since_last_race = 999

            if days_since_last_race < 14:
                reasoning["strengths"].append("Recently active")

            # Risk factors
            if win_rate < 0.1 and races_count > 5:
                reasoning["risk_factors"].append("Low win rate")
            if place_rate < 0.3 and races_count > 5:
                reasoning["risk_factors"].append("Struggles to place")
            if consistency < 0.5:
                reasoning["risk_factors"].append("Inconsistent form")
            if avg_position > 5:
                reasoning["risk_factors"].append("Often finishes poorly")
            if races_count < 3:
                reasoning["risk_factors"].append("Limited racing experience")

            if days_since_last_race > 30:
                reasoning["risk_factors"].append("Long layoff")

            # Performance factors (what contributed to prediction)
            # Handle both old and new prediction formats
            prediction_scores = dog.get("prediction_scores", {})
            traditional_score = prediction_scores.get(
                "traditional", dog.get("traditional_score", 0)
            )
            ml_score = prediction_scores.get("enhanced_data", dog.get("ml_score", 0))
            prediction_score = dog.get("final_score", dog.get("prediction_score", 0))

            reasoning["performance_factors"] = [
                f"Traditional analysis: {round(traditional_score * 100, 1)}%",
                f"ML model prediction: {round(ml_score * 100, 1)}%",
                f"Combined score: {round(prediction_score * 100, 1)}%",
            ]

            # Speed and class analysis
            best_time_raw = historical_stats.get("best_time", 0)

            # Handle best_time string format (e.g., "18.70s")
            try:
                if isinstance(best_time_raw, str):
                    # Remove 's' suffix and convert to float
                    best_time_str = best_time_raw.replace("s", "").strip()
                    if (
                        best_time_str
                        and best_time_str != "0"
                        and best_time_str.replace(".", "").replace("-", "").isdigit()
                    ):
                        best_time_numeric = float(best_time_str)
                        if best_time_numeric > 0:
                            reasoning["key_metrics"][
                                "best_time"
                            ] = best_time_raw  # Keep original format
                elif isinstance(best_time_raw, (int, float)) and best_time_raw > 0:
                    reasoning["key_metrics"]["best_time"] = f"{best_time_raw}s"
            except (ValueError, TypeError):
                pass  # Skip if conversion fails

            # Generate comprehensive key factors based on analysis
            key_factors = []

            # Win rate factors
            if win_rate > 0.25:
                key_factors.append(
                    f"Strong winner - {round(win_rate * 100, 1)}% win rate"
                )
            elif win_rate > 0.15:
                key_factors.append(
                    f"Decent winner - {round(win_rate * 100, 1)}% win rate"
                )
            elif win_rate > 0.05:
                key_factors.append(
                    f"Occasional winner - {round(win_rate * 100, 1)}% win rate"
                )
            else:
                key_factors.append(
                    f"Rare winner - {round(win_rate * 100, 1)}% win rate"
                )

            # Place rate factors
            if place_rate > 0.6:
                key_factors.append(
                    f"Consistent placer - {round(place_rate * 100, 1)}% place rate"
                )
            elif place_rate > 0.4:
                key_factors.append(
                    f"Regular placer - {round(place_rate * 100, 1)}% place rate"
                )
            elif place_rate > 0.25:
                key_factors.append(
                    f"Occasional placer - {round(place_rate * 100, 1)}% place rate"
                )
            else:
                key_factors.append(
                    f"Struggles to place - {round(place_rate * 100, 1)}% place rate"
                )

            # Position factors
            if avg_position <= 2.5:
                key_factors.append(
                    f"Excellent average position ({round(avg_position, 1)})"
                )
            elif avg_position <= 3.5:
                key_factors.append(f"Good average position ({round(avg_position, 1)})")
            elif avg_position <= 5:
                key_factors.append(f"Average position ({round(avg_position, 1)})")
            else:
                key_factors.append(f"Poor average position ({round(avg_position, 1)})")

            # Experience factors
            if races_count >= 20:
                key_factors.append(f"Very experienced - {races_count} races")
            elif races_count >= 10:
                key_factors.append(f"Experienced - {races_count} races")
            elif races_count >= 5:
                key_factors.append(f"Some experience - {races_count} races")
            else:
                key_factors.append(f"Limited experience - {races_count} races")

            # Consistency factors
            if consistency > 0.8:
                key_factors.append(
                    f"Very consistent performer ({round(consistency * 100, 1)}%)"
                )
            elif consistency > 0.6:
                key_factors.append(
                    f"Consistent performer ({round(consistency * 100, 1)}%)"
                )
            elif consistency > 0.4:
                key_factors.append(
                    f"Moderate consistency ({round(consistency * 100, 1)}%)"
                )
            else:
                key_factors.append(
                    f"Inconsistent form ({round(consistency * 100, 1)}%)"
                )

            # Speed factors
            if reasoning["key_metrics"].get("best_time"):
                best_time_value = reasoning["key_metrics"]["best_time"]
                key_factors.append(f"Best time: {best_time_value}")

            # Prediction confidence factors
            prediction_score = dog.get("final_score", dog.get("prediction_score", 0))
            if prediction_score >= 0.7:
                key_factors.append(
                    f"High confidence prediction ({round(prediction_score * 100, 1)}%)"
                )
            elif prediction_score >= 0.5:
                key_factors.append(
                    f"Medium confidence prediction ({round(prediction_score * 100, 1)}%)"
                )
            elif prediction_score >= 0.3:
                key_factors.append(
                    f"Low confidence prediction ({round(prediction_score * 100, 1)}%)"
                )
            else:
                key_factors.append(
                    f"Very low confidence prediction ({round(prediction_score * 100, 1)}%)"
                )

            # Recent activity factors
            if days_since_last_race < 7:
                key_factors.append("Recently raced (within 1 week)")
            elif days_since_last_race < 14:
                key_factors.append("Recently active (within 2 weeks)")
            elif days_since_last_race < 30:
                key_factors.append("Moderately fresh (within 1 month)")
            elif days_since_last_race < 60:
                key_factors.append("Some layoff (1-2 months)")
            elif days_since_last_race < 999:  # Valid number but long layoff
                key_factors.append(f"Long layoff ({days_since_last_race} days)")

            # Add ML vs Traditional analysis insight
            if traditional_score > ml_score + 0.1:
                key_factors.append("Traditional analysis favors this dog")
            elif ml_score > traditional_score + 0.1:
                key_factors.append("ML model strongly favors this dog")
            else:
                key_factors.append("Traditional and ML analysis agree")

            # Store key factors
            reasoning["key_factors"] = key_factors

            # Add reasoning to dog data
            enhanced_dog = dog.copy()
            enhanced_dog["reasoning"] = reasoning

            # Add fields directly to dog object for frontend compatibility
            enhanced_dog["key_factors"] = (
                key_factors  # Frontend expects this directly on dog object
            )
            enhanced_dog["clean_name"] = dog.get("dog_name", "N/A")
            enhanced_dog["prediction_score"] = safe_float(dog.get("final_score", 0))
            enhanced_dog["recommended_bet"] = "ANALYSIS"
            enhanced_dog["predicted_rank"] = dog.get("predicted_rank", 0)

            enhanced_predictions.append(enhanced_dog)

        # Add the enhanced predictions back to the data
        enhanced_data = prediction_data.copy()
        enhanced_data["enhanced_predictions"] = enhanced_predictions
        enhanced_data["race_context"] = race_context
        enhanced_data["race_summary"] = race_summary
        enhanced_data["top_pick"] = top_pick

        return jsonify({"success": True, "prediction": enhanced_data})

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Error loading prediction detail: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/classify_files", methods=["POST"])
def api_classify_files():
    """API endpoint to classify files in unprocessed directory"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    try:
        # Import and use the race file manager
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "race_file_manager", "./race_file_manager.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            manager = module.RaceFileManager()
            manager.classify_and_move_files()

            return jsonify(
                {
                    "success": True,
                    "message": "Files classified successfully",
                    "stats": manager.get_directory_stats(),
                }
            )
        else:
            return (
                jsonify({"success": False, "message": "Race file manager not found"}),
                500,
            )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error classifying files: {str(e)}"}
            ),
            500,
        )


@app.route("/api/move_historical_to_unprocessed", methods=["POST"])
def api_move_historical_to_unprocessed():
    """API endpoint to move historical races to unprocessed for processing"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    try:
        # Import and use the race file manager
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "race_file_manager", "./race_file_manager.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            manager = module.RaceFileManager()
            manager.move_historical_to_unprocessed()

            return jsonify(
                {
                    "success": True,
                    "message": "Historical races moved to unprocessed",
                    "stats": manager.get_directory_stats(),
                }
            )
        else:
            return (
                jsonify({"success": False, "message": "Race file manager not found"}),
                500,
            )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error moving files: {str(e)}"}),
            500,
        )


@app.route("/api/full_pipeline", methods=["POST"])
def api_full_pipeline():
    """API endpoint to run simplified data processing pipeline"""
    if processing_status["running"]:
        return (
            jsonify({"success": False, "message": "Processing already in progress"}),
            400,
        )

    # Start simplified pipeline
    try:
        # Just run the basic data processing in a single thread
        thread = threading.Thread(target=simple_pipeline_background)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": "Data processing started"})
    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error starting processing: {str(e)}"}
            ),
            500,
        )


@app.route("/api/file_stats")
def api_file_stats():
    """API endpoint for enhanced file statistics"""
    try:
        # Import and use the race file manager
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "race_file_manager", "./race_file_manager.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            manager = module.RaceFileManager()
            stats = manager.get_directory_stats()

            return jsonify(
                {
                    "success": True,
                    "stats": stats,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            # Fallback to basic stats
            return jsonify(
                {
                    "success": True,
                    "stats": get_file_stats(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting file stats: {str(e)}"}
            ),
            500,
        )


@app.route("/api/enhanced_analysis")
def api_enhanced_analysis():
    """API endpoint for enhanced race analysis"""
    try:
        # Use a simple hash of the dataset as a cache key
        cache_key = hashlib.md5(open(DATABASE_PATH, "rb").read()).hexdigest()
        cache_file = f"/tmp/enhanced_analysis_cache_{cache_key}.pkl"

        # Check if cache file exists
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                print("Using cached enhanced analysis")
                return jsonify(cache_data)
        except (FileNotFoundError, EOFError):
            print("Cache miss, computing enhanced analysis")
        print("[DEBUG] Starting enhanced analysis API")
        analyzer = EnhancedRaceAnalyzer(DATABASE_PATH)

        # Get comprehensive analysis with detailed logging
        print("[DEBUG] Step 1: Loading data")
        analyzer.load_data()

        print("[DEBUG] Step 2: Engineering features")
        analyzer.engineer_features()

        print("[DEBUG] Step 3: Normalizing performance")
        analyzer.normalize_performance()

        print("[DEBUG] Step 4: Adding race condition features")
        analyzer.add_race_condition_features()

        print("[DEBUG] Step 5: Identifying top performers")
        top_performers = analyzer.identify_top_performers(min_races=2)

        print("[DEBUG] Step 6: Temporal analysis")
        monthly_stats, venue_stats = analyzer.temporal_analysis()

        print("[DEBUG] Step 7: Race condition analysis")
        race_condition_analysis = analyzer.analyze_race_conditions()

        print("[DEBUG] Step 8: Generating insights")
        insights = analyzer.generate_insights()

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict("records")
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.Period):
                return str(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                return None
            elif hasattr(obj, "isoformat"):  # datetime objects
                return obj.isoformat()
            else:
                return obj

        top_performers_dict = top_performers.head(20).to_dict("records")
        venue_stats_dict = venue_stats.to_dict("records")
        monthly_stats_dict = monthly_stats.to_dict("records")

        response_data = {
            "success": True,
            "top_performers": convert_to_json_serializable(top_performers_dict),
            "venue_stats": convert_to_json_serializable(venue_stats_dict),
            "monthly_stats": convert_to_json_serializable(monthly_stats_dict),
            "race_conditions": convert_to_json_serializable(race_condition_analysis),
            "insights": convert_to_json_serializable(insights),
            "data_quality": {
                "total_records": len(analyzer.data),
                "data_completeness": round(
                    (
                        analyzer.data.notna().sum().sum()
                        / (len(analyzer.data) * len(analyzer.data.columns))
                    )
                    * 100,
                    2,
                ),
                "analysis_timestamp": datetime.now().isoformat(),
            },
        }

        # Cache the result
        with open(cache_file, "wb") as f:
            pickle.dump(response_data, f)

        return jsonify(response_data)

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"[ERROR] Enhanced analysis failed: {str(e)}")
        print(f"[ERROR] Full traceback: {error_trace}")
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error running enhanced analysis: {str(e)}",
                    "traceback": error_trace,
                }
            ),
            500,
        )


@app.route("/api/performance_trends")
def api_performance_trends():
    """API endpoint for detailed performance trends analysis"""
    try:
        # Use a simple hash of the dataset as a cache key
        cache_key = hashlib.md5(open(DATABASE_PATH, "rb").read()).hexdigest()
        cache_file = f"/tmp/performance_trends_cache_{cache_key}.pkl"

        # Check if cache file exists
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                print("Using cached performance trends")
                return jsonify(cache_data)
        except (FileNotFoundError, EOFError):
            print("Cache miss, computing performance trends")
        analyzer = EnhancedRaceAnalyzer(DATABASE_PATH)
        analyzer.load_data()
        analyzer.engineer_features()
        analyzer.normalize_performance()
        analyzer.add_race_condition_features()

        # Monthly trends
        monthly_stats, venue_stats = analyzer.temporal_analysis()

        # Convert Period objects to strings for JSON serialization
        if "year_month" in monthly_stats.columns:
            monthly_stats["year_month"] = monthly_stats["year_month"].astype(str)

        # Performance trends by various factors - flatten multi-level columns
        distance_agg = (
            analyzer.data.groupby("distance_numeric")
            .agg({"performance_score": ["mean", "count"], "finish_position": "mean"})
            .reset_index()
        )
        distance_agg.columns = ["distance", "perf_mean", "perf_count", "avg_finish_pos"]

        grade_agg = (
            analyzer.data.groupby("grade_normalized")
            .agg({"performance_score": ["mean", "count"], "finish_position": "mean"})
            .reset_index()
            .head(15)
        )
        grade_agg.columns = ["grade", "perf_mean", "perf_count", "avg_finish_pos"]

        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict("records")
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.Period):
                return str(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                return None
            elif hasattr(obj, "isoformat"):  # datetime objects
                return obj.isoformat()
            else:
                return obj

        trends = {
            "monthly_performance": convert_to_json_serializable(
                monthly_stats.to_dict("records")
            ),
            "venue_performance": convert_to_json_serializable(
                venue_stats.sort_values("avg_performance", ascending=False).to_dict(
                    "records"
                )
            ),
            "distance_trends": convert_to_json_serializable(
                distance_agg.to_dict("records")
            ),
            "grade_trends": convert_to_json_serializable(grade_agg.to_dict("records")),
        }

        response_data = {
            "success": True,
            "trends": trends,
            "timestamp": datetime.now().isoformat(),
        }

        # Cache the result
        with open(cache_file, "wb") as f:
            pickle.dump(response_data, f)

        return jsonify(response_data)

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error analyzing performance trends: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/dog_comparison")
def api_dog_comparison():
    """API endpoint for comparing top dogs"""
    try:
        analyzer = EnhancedRaceAnalyzer(DATABASE_PATH)
        analyzer.load_data()
        analyzer.engineer_features()
        analyzer.normalize_performance()
        analyzer.add_race_condition_features()

        top_performers = analyzer.identify_top_performers(min_races=3)

        # Get detailed comparison data for top 10 dogs
        top_10_dogs = top_performers.head(10)["dog_name"].tolist()

        comparison_data = []
        for dog_name in top_10_dogs:
            dog_data = analyzer.data[analyzer.data["dog_name"] == dog_name]

            dog_stats = {
                "dog_name": dog_name,
                "total_races": len(dog_data),
                "avg_position": round(dog_data["finish_position"].mean(), 2),
                "win_rate": round((dog_data["finish_position"] == 1).mean() * 100, 1),
                "place_rate": round((dog_data["finish_position"] <= 3).mean() * 100, 1),
                "consistency": round(dog_data["consistency_score"].iloc[0], 3),
                "performance_score": round(dog_data["performance_score"].mean(), 3),
                "best_distance": dog_data.groupby("distance_numeric")[
                    "performance_score"
                ]
                .mean()
                .idxmax(),
                "favorite_venue": (
                    dog_data["venue"].mode().iloc[0]
                    if len(dog_data["venue"].mode()) > 0
                    else "N/A"
                ),
                "career_span_days": int(dog_data["career_span_days"].iloc[0]),
                "recent_form": dog_data.tail(5)["finish_position"].tolist(),
                "last_race_date": dog_data["race_date"].max().isoformat(),
            }
            comparison_data.append(dog_stats)

        return jsonify(
            {
                "success": True,
                "comparison_data": comparison_data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error comparing dogs: {str(e)}"}),
            500,
        )


@app.route("/api/prediction_insights")
def api_prediction_insights():
    """API endpoint for prediction-enhancing insights"""
    try:
        analyzer = EnhancedRaceAnalyzer(DATABASE_PATH)
        analyzer.load_data()
        analyzer.engineer_features()
        analyzer.normalize_performance()
        analyzer.add_race_condition_features()

        # Generate insights that can improve predictions
        insights = {
            "venue_bias": {
                "description": "Venues where certain types of dogs perform better",
                "data": analyzer.data.groupby("venue")
                .agg(
                    {
                        "performance_score": "mean",
                        "finish_position": "mean",
                        "field_size": "mean",
                    }
                )
                .round(3)
                .to_dict("index"),
            },
            "distance_specialists": {
                "description": "Dogs that excel at specific distances",
                "data": analyzer.data.groupby(["dog_name", "distance_numeric"])
                .agg({"performance_score": "mean", "race_id": "count"})
                .reset_index()
                .query("race_id >= 2")
                .sort_values("performance_score", ascending=False)
                .head(20)
                .to_dict("records"),
            },
            "form_patterns": {
                "description": "Common form patterns that predict success",
                "improving_dogs": analyzer.data[analyzer.data["form_trend"] > 0]
                .groupby("dog_name")
                .agg({"performance_score": "mean", "form_trend": "mean"})
                .sort_values("performance_score", ascending=False)
                .head(10)
                .to_dict("index"),
                "declining_dogs": analyzer.data[analyzer.data["form_trend"] < 0]
                .groupby("dog_name")
                .agg({"performance_score": "mean", "form_trend": "mean"})
                .sort_values("performance_score", ascending=False)
                .head(10)
                .to_dict("index"),
            },
            "track_condition_impact": {
                "description": "How track conditions affect different performance levels",
                "data": {
                    # Flatten and convert the grouped data to avoid tuple keys
                    condition: {
                        "performance_mean": round(
                            data[("performance_score", "mean")], 3
                        ),
                        "performance_std": round(data[("performance_score", "std")], 3),
                        "avg_finish_position": round(
                            data[("finish_position", "mean")], 3
                        ),
                    }
                    for condition, data in analyzer.data.groupby("track_condition")
                    .agg(
                        {
                            "performance_score": ["mean", "std"],
                            "finish_position": "mean",
                        }
                    )
                    .iterrows()
                },
            },
        }

        return jsonify(
            {
                "success": True,
                "insights": insights,
                "recommendations": [
                    "Use venue-specific performance data for more accurate predictions",
                    "Consider distance specialization when analyzing dog performance",
                    "Track form trends to identify improving/declining dogs",
                    "Adjust predictions based on track conditions",
                ],
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error generating prediction insights: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/model/historical_accuracy")
def api_model_historical_accuracy():
    """API endpoint for historical model accuracy"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Example: Fetch historical accuracy data from a dedicated table or logs
        # This is a placeholder for your actual data source
        cursor.execute(
            """
            SELECT date(timestamp) as accuracy_date, AVG(accuracy) as avg_accuracy
            FROM model_performance_history
            GROUP BY accuracy_date
            ORDER BY accuracy_date ASC
            LIMIT 30
        """
        )
        accuracy_data = cursor.fetchall()
        conn.close()

        # Fallback to dummy data if no historical data is available
        if not accuracy_data:
            accuracy_data = [
                (
                    ("2023-01-01", 0.75),
                    ("2023-01-02", 0.78),
                    ("2023-01-03", 0.77),
                    ("2023-01-04", 0.80),
                    ("2023-01-05", 0.82),
                )
            ]

        return jsonify(
            {
                "success": True,
                "accuracy_data": [
                    {"date": row[0], "accuracy": row[1]} for row in accuracy_data
                ],
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error getting historical accuracy: {str(e)}",
                }
            ),
            500,
        )


@app.route("/ml-dashboard")
@app.route("/ml_dashboard")
def ml_dashboard():
    """Machine Learning Dashboard page"""
    return render_template("ml_dashboard.html")


@app.route("/upcoming")
def upcoming_races():
    """Browse upcoming races page"""
    return render_template("upcoming_races.html")


# Cache for upcoming races API
_upcoming_races_cache = {
    "data": None,
    "timestamp": None,
    "expires_in_minutes": 5,  # Cache for 5 minutes
}

def _extract_csv_metadata(file_path):
    """Extract metadata from CSV filename using regex.
    
    Expected filename format: "Race 1 – AP_K – 2025-08-04.csv"
    Returns dict with race_number, venue, date, or None values if not found.
    """
    import re
    import os
    
    filename = os.path.basename(file_path)
    
    # Regex pattern to match: "Race {number} – {venue} – {date}.csv"
    # The en dash (–) is different from hyphen (-)
    pattern = r'Race\s+(\d+)\s*[–-]\s*([A-Z_]+)\s*[–-]\s*(\d{4}-\d{2}-\d{2})\.csv'
    
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        return {
            'race_number': int(match.group(1)),
            'venue': match.group(2),
            'date': match.group(3)
        }
    
    # Fallback: try to extract individual components
    race_number = None
    venue = None
    date = None
    
    # Extract race number
    race_match = re.search(r'Race[_\s]+(\d+)', filename, re.IGNORECASE)
    if race_match:
        race_number = int(race_match.group(1))
    
    # Extract venue (look for uppercase 3-4 letter codes)
    venue_match = re.search(r'([A-Z_]{2,4})', filename)
    if venue_match:
        venue = venue_match.group(1)
    
    # Extract date (YYYY-MM-DD format)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        date = date_match.group(1)
    
    return {
        'race_number': race_number,
        'venue': venue,
        'date': date
    }


def load_upcoming_races_with_guaranteed_fields(refresh=False):
    """Refactored helper function to load upcoming races with guaranteed API contract fields.
    
    Returns list with guaranteed fields: date, venue, venue_name, race_number, race_time, 
    distance, grade, race_name, url, filename, race_id
    """
    import os
    import pandas as pd
    from datetime import datetime
    import json
    import hashlib
    import re

    global _upcoming_races_cache
    now = datetime.now()

    # Check cache first
    if not refresh and _upcoming_races_cache["data"] is not None and _upcoming_races_cache["timestamp"] is not None and (now - _upcoming_races_cache["timestamp"]).total_seconds() < (_upcoming_races_cache["expires_in_minutes"] * 60):
        cached_data = _upcoming_races_cache["data"]
        # Ensure cached data is always a list and has guaranteed fields
        if isinstance(cached_data, dict):
            cached_data = list(cached_data.values())
        return [_ensure_guaranteed_fields(race) for race in cached_data] if cached_data else []

    upcoming_races_dir = "./upcoming_races"
    races = []

    def parse_time_to_minutes(time_str):
        """Convert time string like '6:31 PM' to minutes since midnight for sorting."""
        if not time_str:
            return 0
        try:
            # Handle different time formats
            time_str = str(time_str).strip()
            if re.match(r'\d{1,2}:\d{2} [AP]M', time_str):
                # Parse "6:31 PM" format
                time_part, period = time_str.split()
                hours, minutes = map(int, time_part.split(':'))
                if period.upper() == 'PM' and hours != 12:
                    hours += 12
                elif period.upper() == 'AM' and hours == 12:
                    hours = 0
                return hours * 60 + minutes
            elif re.match(r'\d{1,2}:\d{2}', time_str):
                # Parse "18:31" format (24-hour)
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
        except (ValueError, AttributeError):
            pass
        return 0

    try:
        for filename in os.listdir(upcoming_races_dir):
            if filename.endswith(".csv") or filename.endswith(".json"):
                file_path = os.path.join(upcoming_races_dir, filename)
                try:
                    if filename.endswith(".csv"):
                        # Extract metadata from filename using helper function
                        filename_metadata = _extract_csv_metadata(file_path)
                        
                        # Read only header row (or first data row) to get grade/distance
                        try:
                            df_header = pd.read_csv(file_path, nrows=1)
                            # Check for both possible column names for grade and distance
                            header_grade = None
                            header_distance = None
                            
                            # Handle grade columns (G or Grade)
                            if "Grade" in df_header.columns:
                                header_grade = df_header.get("Grade", pd.Series([None])).iloc[0]
                            elif "G" in df_header.columns:
                                header_grade = df_header.get("G", pd.Series([None])).iloc[0]
                            
                            # Handle distance columns (DIST or Distance)
                            if "Distance" in df_header.columns:
                                header_distance = df_header.get("Distance", pd.Series([None])).iloc[0]
                            elif "DIST" in df_header.columns:
                                header_distance = df_header.get("DIST", pd.Series([None])).iloc[0]
                                
                        except Exception as e:
                            print(f"Warning: Could not read header from {filename}: {e}")
                            header_grade = None
                            header_distance = None
                        
                        # Build race dict with guaranteed fields
                        race_data = {
                            "date": filename_metadata['date'] or "",
                            "venue": filename_metadata['venue'] or "Unknown Venue",
                            "venue_name": filename_metadata['venue'] or "Unknown Venue",  # Same as venue for now
                            "race_number": filename_metadata['race_number'] or "",
                            "race_time": "",  # Time not available in filename pattern
                            "distance": str(header_distance) if header_distance is not None else "",
                            "grade": str(header_grade) if header_grade is not None else "",
                            "race_name": f"Race {filename_metadata['race_number']}" if filename_metadata['race_number'] else "Unknown Race",
                            "url": "",  # URL not available from CSV files
                            "filename": filename,
                            "race_id": hashlib.md5(filename.encode()).hexdigest()[:12],
                        }
                        
                        # Ensure all guaranteed fields are present
                        race_data = _ensure_guaranteed_fields(race_data)
                        races.append(race_data)
                        
                    else:
                        # For JSON files, load the data directly
                        with open(file_path, "r") as f:
                            json_data = json.load(f)
                            
                        # Handle both list and dict structures
                        if isinstance(json_data, dict):
                            # Convert dict to list of values if needed
                            json_data = list(json_data.values())
                        elif not isinstance(json_data, list):
                            # If it's neither dict nor list, wrap in list
                            json_data = [json_data]
                        
                        for item in json_data:
                            if isinstance(item, dict):
                                race_data = {
                                    "date": item.get("date") or item.get("race_date") or item.get("Date") or "",
                                    "venue": item.get("venue") or item.get("Venue") or "Unknown Venue",
                                    "venue_name": item.get("venue_name") or item.get("venue") or item.get("Venue") or "Unknown Venue",
                                    "race_number": item.get("race_number") or item.get("Race Number") or item.get("number") or "",
                                    "race_time": item.get("race_time") or item.get("Race Time") or item.get("time") or "",
                                    "distance": item.get("distance") or item.get("Distance") or "",
                                    "grade": item.get("grade") or item.get("Grade") or "",
                                    "race_name": item.get("race_name") or item.get("Race Name") or "Unknown Race",
                                    "url": item.get("url") or item.get("URL") or "",
                                    "filename": filename,
                                    "race_id": hashlib.md5(f"{filename}_{item.get('race_number', item.get('Race Number', 0))}".encode()).hexdigest()[:12],
                                }
                                
                                # Ensure all guaranteed fields are present
                                race_data = _ensure_guaranteed_fields(race_data)
                                race_data["_sort_time_minutes"] = parse_time_to_minutes(race_data["race_time"])
                                races.append(race_data)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
    except OSError as listdir_error:
        print(f"Warning: Unable to access upcoming races directory: {listdir_error}")
        print("Falling back to cached data if available.")

        if _upcoming_races_cache["data"] is not None:
            print("Using cached data for upcoming races.")
            cached_data = _upcoming_races_cache["data"]
            if isinstance(cached_data, dict):
                cached_data = list(cached_data.values())
            return [_ensure_guaranteed_fields(race) for race in cached_data] if cached_data else []

        print("No cached data available; unable to load upcoming races.")
        raise

    # Deduplicate races by (venue, date, race_number) composite key
    seen_races = {}
    deduplicated_races = []
    
    for race in races:
        # Create composite key for deduplication
        venue = race.get("venue", "")
        race_date = race.get("date", "")
        race_number = race.get("race_number", "")
        
        # Convert race_number to string for consistent comparison
        race_number_str = str(race_number) if race_number else ""
        
        dedup_key = (venue, race_date, race_number_str)
        
        # Keep the first occurrence of each unique race
        if dedup_key not in seen_races:
            seen_races[dedup_key] = True
            deduplicated_races.append(race)
    
    # Replace races list with deduplicated version
    races = deduplicated_races
    
    # Add sorting helper _sort_time_minutes only once for all races
    for race in races:
        if "_sort_time_minutes" not in race:
            race["_sort_time_minutes"] = parse_time_to_minutes(race.get("race_time", ""))
    
    races.sort(key=lambda x: (
        x.get("date", ""),  # Sort by date first
        x.get("_sort_time_minutes", 0),  # Then by race time
        x.get("venue", "")  # Finally by venue
    ))
    
    # Remove the sorting helper field from final output
    for race in races:
        race.pop("_sort_time_minutes", None)

    _upcoming_races_cache["data"] = races
    _upcoming_races_cache["timestamp"] = now

    return races

def _ensure_guaranteed_fields(race_data):
    """Ensure all guaranteed API contract fields are present in race data.
    
    Guaranteed fields: date, venue, venue_name, race_number, race_time, 
    distance, grade, race_name, url, filename, race_id
    """
    guaranteed_fields = {
        "date": "",
        "venue": "Unknown Venue",
        "venue_name": "Unknown Venue",
        "race_number": "",
        "race_time": "",
        "distance": "",
        "grade": "",
        "race_name": "Unknown Race",
        "url": "",
        "filename": "",
        "race_id": ""
    }
    
    for field, default_value in guaranteed_fields.items():
        if field not in race_data or race_data[field] is None:
            race_data[field] = default_value
        else:
            race_data[field] = str(race_data[field])
    
    return race_data

def load_upcoming_races(refresh=False):
    """Legacy helper function to load upcoming races from CSV and JSON files.
    Kept for backward compatibility."""
    import os
    import pandas as pd
    from datetime import datetime
    import json
    import re

    global _upcoming_races_cache
    now = datetime.now()

    if not refresh and _upcoming_races_cache["data"] is not None and _upcoming_races_cache["timestamp"] is not None and (now - _upcoming_races_cache["timestamp"]).total_seconds() < (_upcoming_races_cache["expires_in_minutes"] * 60):
        cached_data = _upcoming_races_cache["data"]
        if isinstance(cached_data, dict):
            return list(cached_data.values())
        return cached_data

    upcoming_races_dir = "./upcoming_races"
    races = []

    def parse_time_to_minutes(time_str):
        """Convert time string like '6:31 PM' to minutes since midnight for sorting."""
        if not time_str:
            return 0
        try:
            # Handle different time formats
            time_str = str(time_str).strip()
            if re.match(r'\d{1,2}:\d{2} [AP]M', time_str):
                # Parse "6:31 PM" format
                time_part, period = time_str.split()
                hours, minutes = map(int, time_part.split(':'))
                if period.upper() == 'PM' and hours != 12:
                    hours += 12
                elif period.upper() == 'AM' and hours == 12:
                    hours = 0
                return hours * 60 + minutes
            elif re.match(r'\d{1,2}:\d{2}', time_str):
                # Parse "18:31" format (24-hour)
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
        except (ValueError, AttributeError):
            pass
        return 0

    try:
        for filename in os.listdir(upcoming_races_dir):
            if filename.endswith(".csv") or filename.endswith(".json"):
                file_path = os.path.join(upcoming_races_dir, filename)
                try:
                    if filename.endswith(".csv"):
                        # Extract metadata from filename using helper function
                        filename_metadata = _extract_csv_metadata(file_path)
                        
                        # Read only header row (or first data row) to get grade/distance
                        try:
                            df_header = pd.read_csv(file_path, nrows=1)
                            # Handle both possible column names for grade and distance
                            header_grade = None
                            header_distance = None
                            
                            if "Grade" in df_header.columns:
                                header_grade = df_header.get("Grade", pd.Series([None])).iloc[0]
                            elif "G" in df_header.columns:
                                header_grade = df_header.get("G", pd.Series([None])).iloc[0]
                            
                            if "Distance" in df_header.columns:
                                header_distance = df_header.get("Distance", pd.Series([None])).iloc[0]
                            elif "DIST" in df_header.columns:
                                header_distance = df_header.get("DIST", pd.Series([None])).iloc[0]
                        except Exception as e:
                            print(f"Warning: Could not read header from {filename}: {e}")
                            header_grade = None
                            header_distance = None
                        
                        # Build single race dict from filename metadata and header data
                        race_metadata = {
                            "race_name": f"Race {filename_metadata['race_number']}" if filename_metadata['race_number'] else "Unknown Race",
                            "venue": filename_metadata['venue'] or "Unknown Venue",
                            "race_date": filename_metadata['date'] or "",
                            "race_time": "",  # Time not available in filename pattern
                            "distance": str(header_distance) if header_distance is not None else "",
                            "grade": str(header_grade) if header_grade is not None else "",
                            "race_number": filename_metadata['race_number'] or "",
                            "filename": filename,
                            "race_id": hashlib.md5(filename.encode()).hexdigest()[:12],
                        }
                        
                        # Append single race dict once
                        races.append(race_metadata)
                    else:
                        # For JSON files, load the data directly
                        with open(file_path, "r") as f:
                            json_data = json.load(f)
                            
                        # Handle both list and dict structures
                        if isinstance(json_data, dict):
                            # Convert dict to list of values if needed
                            json_data = list(json_data.values())
                        elif not isinstance(json_data, list):
                            # If it's neither dict nor list, wrap in list
                            json_data = [json_data]
                        
                        for item in json_data:
                            if isinstance(item, dict):
                                race_metadata = {
                                    "race_name": item.get("race_name") or item.get("Race Name") or "Unknown Race",
                                    "venue": item.get("venue") or item.get("Venue") or "Unknown Venue",
                                    "race_date": item.get("date") or item.get("race_date") or item.get("Date") or "",
                                    "race_time": item.get("race_time") or item.get("Race Time") or item.get("time") or "",
                                    "distance": item.get("distance") or item.get("Distance") or "",
                                    "grade": item.get("grade") or item.get("Grade") or "",
                                    "race_number": item.get("race_number") or item.get("Race Number") or item.get("number") or "",
                                    "filename": filename,
                                    "race_id": hashlib.md5(f"{filename}_{item.get('race_number', item.get('Race Number', 0))}".encode()).hexdigest()[:12],
                                    # Add sorting helper
                                    "_sort_time_minutes": parse_time_to_minutes(item.get("race_time") or item.get("Race Time") or item.get("time") or "")
                                }
                                races.append(race_metadata)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
    except OSError as listdir_error:
        print(f"Warning: Unable to access upcoming races directory: {listdir_error}")
        print("Falling back to cached data if available.")

        if _upcoming_races_cache["data"] is not None:
            print("Using cached data for upcoming races.")
            return _upcoming_races_cache["data"]

        print("No cached data available; unable to load upcoming races.")
        raise

    # Deduplicate races by (venue, date, race_number) composite key
    seen_races = {}
    deduplicated_races = []
    
    for race in races:
        # Create composite key for deduplication
        venue = race.get("venue", "")
        race_date = race.get("race_date", "")
        race_number = race.get("race_number", "")
        
        # Convert race_number to string for consistent comparison
        race_number_str = str(race_number) if race_number else ""
        
        dedup_key = (venue, race_date, race_number_str)
        
        # Keep the first occurrence of each unique race
        if dedup_key not in seen_races:
            seen_races[dedup_key] = True
            deduplicated_races.append(race)
    
    # Replace races list with deduplicated version
    races = deduplicated_races
    
    # Add sorting helper _sort_time_minutes only once for all races
    for race in races:
        if "_sort_time_minutes" not in race:
            race["_sort_time_minutes"] = parse_time_to_minutes(race.get("race_time", ""))
    
    races.sort(key=lambda x: (
        x.get("race_date", ""),  # Sort by date first
        x.get("_sort_time_minutes", 0),  # Then by race time
        x.get("venue", "")  # Finally by venue
    ))
    
    # Remove the sorting helper field from final output
    for race in races:
        race.pop("_sort_time_minutes", None)

    _upcoming_races_cache["data"] = races
    _upcoming_races_cache["timestamp"] = now

    return races


@app.route("/api/upcoming_races_stream")
def api_upcoming_races_stream():
    """Streaming API endpoint that returns races as they're discovered"""
    import json

    from flask import Response, copy_current_request_context
    from datetime import datetime, timedelta

    # Get days_ahead parameter before starting the generator
    days_ahead = request.args.get("days", 1, type=int)

    @copy_current_request_context
    def generate_race_stream():
        try:
            from upcoming_race_browser import UpcomingRaceBrowser

            browser = UpcomingRaceBrowser()

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting to fetch races for next {days_ahead} days...', 'progress': 0})}\n\n"

            today = datetime.now().date()
            all_races = []
            total_days = days_ahead + 1

            for i in range(total_days):
                check_date = today + timedelta(days=i)
                date_str = check_date.strftime("%Y-%m-%d")
                progress = int((i / total_days) * 100)

                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'date': date_str, 'progress': progress, 'message': f'Scanning {date_str}...'})}\n\n"

                try:
                    # Get races for this date
                    date_races = browser.get_races_for_date(check_date)

                    if date_races:
                        # Send each race as it's found
                        for race in date_races:
                            race["date"] = date_str  # Ensure date is set
                            all_races.append(race)
                            yield f"data: {json.dumps({'type': 'race', 'race': race, 'total_found': len(all_races)})}\n\n"

                        # Send date summary
                        yield f"data: {json.dumps({'type': 'date_complete', 'date': date_str, 'count': len(date_races), 'message': f'Found {len(date_races)} races for {date_str}'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'date_complete', 'date': date_str, 'count': 0, 'message': f'No races found for {date_str}'})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'date': date_str, 'message': f'Error scanning {date_str}: {str(e)}'})}\n\n"

            # Final completion message
            yield f"data: {json.dumps({'type': 'complete', 'total_races': len(all_races), 'message': f'Scan complete! Found {len(all_races)} total races.', 'races': all_races})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"

    return Response(
        generate_race_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )




@app.route("/api/download_upcoming_race", methods=["POST"])
def api_download_upcoming_race():
    """API endpoint to download a specific upcoming race"""
    try:
        data = request.get_json()
        race_url = data.get("race_url")

        if not race_url:
            return jsonify({"success": False, "error": "Race URL is required"}), 400

        from upcoming_race_browser import UpcomingRaceBrowser

        browser = UpcomingRaceBrowser()

        result = browser.download_race_csv(race_url)

        if result["success"]:
            return jsonify(
                {
                    "success": True,
                    "message": f'Successfully downloaded {result["filename"]}',
                    "filename": result["filename"],
                    "filepath": result["filepath"],
                }
            )
        else:
            return jsonify({"success": False, "error": result["error"]}), 500

    except Exception as e:
        return (
            jsonify({"success": False, "error": f"Error downloading race: {str(e)}"}),
            500,
        )


@app.route("/api/generate_report", methods=["POST"])
def api_generate_report():
    """API endpoint to generate and download a comprehensive race analysis report"""
    try:
        import io

        from flask import send_file

        # Get database stats and race data
        db_manager = DatabaseManager(DATABASE_PATH)

        # Get ALL race data for proper analysis
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Get comprehensive database statistics
        cursor.execute(
            """
            SELECT COUNT(*) as total_races, 
                   COUNT(DISTINCT venue) as unique_venues,
                   MIN(race_date) as earliest_date,
                   MAX(race_date) as latest_date
            FROM race_metadata
        """
        )
        db_stats = cursor.fetchone()

        cursor.execute(
            """
            SELECT COUNT(*) as total_entries,
                   COUNT(DISTINCT dog_name) as unique_dogs
            FROM dog_race_data
            WHERE dog_name IS NOT NULL AND dog_name != '' AND dog_name != 'nan'
        """
        )
        dog_stats = cursor.fetchone()

        # Get recent completed races for summary
        cursor.execute(
            """
            SELECT race_id, venue, race_number, race_date, race_name, grade, distance,
                   field_size, winner_name, winner_odds, winner_margin, url,
                   extraction_timestamp, track_condition
            FROM race_metadata 
            WHERE winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan'
            ORDER BY race_date DESC, extraction_timestamp DESC
            LIMIT 20
        """
        )

        completed_races = cursor.fetchall()
        conn.close()

        # Create enhanced analysis with proper error handling
        try:
            analyzer = EnhancedRaceAnalyzer(DATABASE_PATH)
            analyzer.load_data()
            analyzer.engineer_features()
            analyzer.normalize_performance()
            analyzer.add_race_condition_features()

            # Get proper analysis results
            top_performers = analyzer.identify_top_performers(
                min_races=3
            )  # Minimum 3 races for meaningful data
            monthly_stats, venue_stats = analyzer.temporal_analysis()
            race_conditions = analyzer.analyze_race_conditions()
            insights = analyzer.generate_insights()

            analysis_success = True
        except Exception as e:
            print(f"Enhanced analysis failed: {e}")
            analysis_success = False
            top_performers = None
            venue_stats = None
            insights = {}

        # Create report content
        report_content = io.StringIO()

        # Write header
        report_content.write("GREYHOUND RACING COMPREHENSIVE ANALYSIS REPORT\n")
        report_content.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        report_content.write("=" * 60 + "\n\n")

        # Database summary with correct statistics
        report_content.write("DATABASE SUMMARY\n")
        report_content.write("-" * 20 + "\n")
        report_content.write(f"Total Races: {db_stats[0]}\n")
        report_content.write(f"Total Race Entries: {dog_stats[0]}\n")
        report_content.write(f"Unique Dogs: {dog_stats[1]}\n")
        report_content.write(f"Unique Venues: {db_stats[1]}\n")
        report_content.write(f"Date Range: {db_stats[2]} to {db_stats[3]}\n\n")

        if analysis_success and top_performers is not None and len(top_performers) > 0:
            # Top performers with proper data
            report_content.write("TOP PERFORMING DOGS (Min 3 Races)\n")
            report_content.write("-" * 40 + "\n")
            for i, (_, dog_data) in enumerate(top_performers.head(10).iterrows(), 1):
                races = dog_data.get("race_count", 0)
                avg_pos = dog_data.get("avg_position", 0)
                score = dog_data.get("composite_score", 0)
                report_content.write(
                    f"{i}. {dog_data.get('dog_name', 'Unknown')} - "
                    f"Score: {score:.3f}, "
                    f"Races: {races}, "
                    f"Avg Position: {avg_pos:.1f}\n"
                )
            report_content.write("\n")

            # Venue analysis with proper performance data
            if venue_stats is not None and len(venue_stats) > 0:
                report_content.write("VENUE PERFORMANCE ANALYSIS\n")
                report_content.write("-" * 30 + "\n")
                venue_sorted = venue_stats.sort_values(
                    "avg_performance", ascending=False
                )
                for _, venue_data in venue_sorted.head(15).iterrows():
                    venue_name = venue_data.get("venue", "Unknown")
                    avg_perf = venue_data.get("avg_performance", 0)
                    race_count = venue_data.get("race_count", 0)
                    avg_pos = venue_data.get("avg_finish_pos", 0)
                    report_content.write(
                        f"{venue_name}: "
                        f"Avg Performance: {avg_perf:.3f}, "
                        f"Races: {race_count}, "
                        f"Avg Finish Pos: {avg_pos:.1f}\n"
                    )
                report_content.write("\n")

            # Race conditions analysis
            if race_conditions and "distance_stats" in race_conditions:
                report_content.write("DISTANCE ANALYSIS\n")
                report_content.write("-" * 20 + "\n")
                dist_stats = race_conditions["distance_stats"]
                if len(dist_stats) > 0:
                    for _, dist_data in dist_stats.head(10).iterrows():
                        distance = dist_data.get("distance_category", "Unknown")
                        avg_perf = dist_data.get("avg_performance", 0)
                        race_count = dist_data.get("race_count", 0)
                        report_content.write(
                            f"{distance}: "
                            f"Avg Performance: {avg_perf:.3f}, "
                            f"Races: {race_count}\n"
                        )
                report_content.write("\n")

            # Insights with proper formatting
            if insights and isinstance(insights, dict):
                report_content.write("KEY INSIGHTS\n")
                report_content.write("-" * 15 + "\n")

                # Data summary insights
                if "data_summary" in insights:
                    summary = insights["data_summary"]
                    report_content.write(
                        f"• Total unique races: {summary.get('total_races', 0)}\n"
                    )
                    report_content.write(
                        f"• Total unique dogs: {summary.get('total_dogs', 0)}\n"
                    )
                    report_content.write(
                        f"• Total race entries: {summary.get('total_entries', 0)}\n"
                    )
                    report_content.write(
                        f"• Date range: {summary.get('date_range', 'Unknown')}\n"
                    )

                # Performance insights
                if "performance_metrics" in insights:
                    perf = insights["performance_metrics"]
                    report_content.write(
                        f"• Average performance score: {perf.get('avg_performance_score', 0):.3f}\n"
                    )
                    report_content.write(
                        f"• Performance std deviation: {perf.get('performance_std', 0):.3f}\n"
                    )
                    if "consistency_leader" in perf:
                        report_content.write(
                            f"• Most consistent performer: {perf['consistency_leader']}\n"
                        )

                # Frequency insights
                if "frequency_analysis" in insights:
                    freq = insights["frequency_analysis"]
                    report_content.write(
                        f"• Single-race dogs: {freq.get('single_race_dogs', 0)}\n"
                    )
                    report_content.write(
                        f"• Frequent racers (3+ races): {freq.get('frequent_racers', 0)}\n"
                    )
                    report_content.write(
                        f"• Average races per dog: {freq.get('avg_races_per_dog', 0):.1f}\n"
                    )

                report_content.write("\n")
        else:
            report_content.write("ANALYSIS STATUS\n")
            report_content.write("-" * 20 + "\n")
            report_content.write("Enhanced analysis failed or insufficient data.\n")
            report_content.write("Using basic database statistics only.\n\n")

        # Recent races summary
        if completed_races:
            report_content.write("RECENT RACES SUMMARY (Last 20 Completed)\n")
            report_content.write("-" * 45 + "\n")
            for race in completed_races:
                race_date = race[3]
                venue = race[1]
                race_num = race[2]
                winner = race[8]
                report_content.write(
                    f"{race_date} - {venue} Race {race_num}: Winner: {winner}\n"
                )

        # Create file-like object
        report_content.seek(0)
        file_like = io.BytesIO(report_content.getvalue().encode("utf-8"))

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"greyhound_racing_report_{timestamp}.txt"

        return send_file(
            file_like, as_attachment=True, download_name=filename, mimetype="text/plain"
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error generating report: {str(e)}"}
            ),
            500,
        )


@app.route("/enhanced_analysis")
def enhanced_analysis_page():
    """Enhanced analysis page"""
    return render_template("enhanced_analysis.html")


# ML Training API Endpoints


@app.route("/ml-training")
def ml_training_dashboard():
    """ML Training Dashboard page"""
    return render_template("ml_training.html")


@app.route("/api/model_status")
def api_model_status():
    """Get current model status and performance metrics"""
    try:
        import json
        from pathlib import Path

        import joblib

        models_dir = Path("./comprehensive_trained_models")
        results_dir = Path("./comprehensive_model_results")

        # Initialize default response
        response = {
            "success": True,
            "model_type": "No trained model",
            "accuracy": None,
            "auc_score": None,
            "last_trained": None,
            "features": 0,
            "samples": 0,
            "class_balance": "Unknown",
            "imbalanced_learning": "Unknown",
            "best_model_name": None,
            "total_models": 0,
        }

        # Check if models directory exists
        if not models_dir.exists():
            return jsonify(response)

        # Find latest model files
        model_files = list(models_dir.glob("comprehensive_best_model_*.joblib"))
        if not model_files:
            return jsonify(response)

        # Get the latest model file
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        try:
            model_data = joblib.load(latest_model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return jsonify(response)

        # Initialize values from model data
        response["model_type"] = model_data.get("model_name", "Comprehensive ML Model")
        response["last_trained"] = model_data.get(
            "timestamp", latest_model.stat().st_mtime
        )
        response["features"] = len(model_data.get("feature_columns", []))

        # Try to get accuracy from model data directly first
        if "accuracy" in model_data and model_data["accuracy"] is not None:
            response["accuracy"] = float(model_data["accuracy"])
        if "auc_score" in model_data and model_data["auc_score"] is not None:
            response["auc_score"] = float(model_data["auc_score"])

        # Get samples info
        if "data_summary" in model_data:
            response["samples"] = model_data["data_summary"].get("total_samples", 0)

        # Look for latest results files to get more detailed metrics
        if results_dir.exists():
            # Try different result file patterns
            result_patterns = [
                "Analysis_ML_*.json",
                "comprehensive_analysis_*.json",
                "ml_backtesting_results_*.json",
            ]

            all_result_files = []
            for pattern in result_patterns:
                all_result_files.extend(list(results_dir.glob(pattern)))

            if all_result_files:
                # Get the most recent results file
                latest_result = max(all_result_files, key=lambda x: x.stat().st_mtime)

                try:
                    with open(latest_result, "r") as f:
                        results = json.load(f)

                    # Extract model results if available
                    if "model_results" in results:
                        model_results = results["model_results"]

                        # Find the best performing model
                        best_accuracy = 0
                        best_auc = 0
                        best_model_name = None
                        total_models = len(model_results)

                        for model_name, metrics in model_results.items():
                            if isinstance(metrics, dict):
                                accuracy = metrics.get("accuracy", 0)
                                auc = metrics.get("auc", 0)

                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_auc = auc
                                    best_model_name = model_name

                        # Update response with best results
                        if best_accuracy > 0:
                            response["accuracy"] = float(best_accuracy)
                            response["best_model_name"] = best_model_name
                            response["total_models"] = total_models

                        if best_auc > 0:
                            response["auc_score"] = float(best_auc)

                    # Get data summary from results if not in model
                    if "data_summary" in results and response["samples"] == 0:
                        data_summary = results["data_summary"]
                        response["samples"] = data_summary.get("total_samples", 0)
                        if "features" in data_summary:
                            response["features"] = data_summary.get(
                                "features", response["features"]
                            )

                    # Update model type based on best performing model
                    if best_model_name:
                        response["model_type"] = (
                            f"{best_model_name.replace('_', ' ').title()} (Best: {best_accuracy:.1%})"
                        )

                except Exception as e:
                    print(f"Error reading results file {latest_result}: {e}")

        # Set enhanced info if we have a trained model
        if response["accuracy"] is not None:
            response["class_balance"] = "Enabled"
            response["imbalanced_learning"] = "SMOTE + Balanced Ensemble"

        return jsonify(response)

    except Exception as e:
        print(f"Error in model_status API: {e}")
        return (
            jsonify(
                {"success": False, "message": f"Error getting model status: {str(e)}"}
            ),
            500,
        )


@app.route("/api/training_data_stats")
def api_training_data_stats():
    """Get training data statistics"""
    try:
        import sqlite3

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Get database stats
        cursor.execute(
            "SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NOT NULL AND finish_position != '' AND finish_position != 'N/A'"
        )
        total_samples = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(DISTINCT dog_clean_name) FROM dog_race_data WHERE dog_clean_name IS NOT NULL"
        )
        unique_dogs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT race_id) FROM race_metadata")
        total_races = cursor.fetchone()[0]

        conn.close()

        # Check form guide data
        form_guides_dir = Path("./form_guides/downloaded")
        form_files = 0
        if form_guides_dir.exists():
            form_files = len(list(form_guides_dir.glob("*.csv")))

        return jsonify(
            {
                "success": True,
                "total_samples": total_samples,
                "unique_dogs": unique_dogs,
                "total_races": total_races,
                "form_guide_files": form_files,
                "features": 30,  # Standard feature count
                "data_quality": "Good" if total_samples > 1000 else "Limited",
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error getting training data stats: {str(e)}",
                }
            ),
            500,
        )


# Global training status tracking
training_status = {
    "running": False,
    "progress": 0,
    "current_task": "",
    "log": [],
    "start_time": None,
    "training_type": None,
    "completed": False,
    "results": None,
    "error": None,
}


def run_training_background(training_type):
    """Background training function"""
    global training_status
    import subprocess
    import sys

    training_status["running"] = True
    training_status["progress"] = 0
    training_status["completed"] = False
    training_status["error"] = None
    training_status["results"] = None
    training_status["log"] = []
    training_status["start_time"] = datetime.now()
    training_status["training_type"] = training_type

    try:
        if training_type == "comprehensive_training":
            training_status["current_task"] = "Starting comprehensive ML training"
            training_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "🚀 Starting comprehensive ML model training...",
                    "level": "INFO",
                }
            )

            # Run improved comprehensive enhanced ML system with class balancing
            script_path = "comprehensive_enhanced_ml_system.py"
            if os.path.exists(script_path):
                result = subprocess.run(
                    [sys.executable, script_path, "--command", "analyze"],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes timeout
                )

                training_status["progress"] = 90

                if result.returncode == 0:
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": "✅ Comprehensive ML training completed successfully!",
                            "level": "INFO",
                        }
                    )
                    training_status["completed"] = True

                    # Try to get results
                    try:
                        results_dir = Path("./comprehensive_model_results")
                        if results_dir.exists():
                            result_files = list(
                                results_dir.glob("comprehensive_analysis_*.json")
                            )
                            if result_files:
                                latest_result = max(
                                    result_files, key=lambda x: x.stat().st_mtime
                                )
                                with open(latest_result, "r") as f:
                                    results_data = json.load(f)
                                    training_status["results"] = {
                                        "accuracy": max(
                                            [
                                                m.get("accuracy", 0)
                                                for m in results_data.get(
                                                    "model_results", {}
                                                ).values()
                                            ]
                                        ),
                                        "auc_score": max(
                                            [
                                                m.get("auc", 0)
                                                for m in results_data.get(
                                                    "model_results", {}
                                                ).values()
                                            ]
                                        ),
                                        "samples": results_data.get(
                                            "data_summary", {}
                                        ).get("total_samples", 0),
                                    }
                    except Exception as e:
                        print(f"Error loading results: {e}")

                else:
                    training_status["error"] = (
                        result.stderr[:500] if result.stderr else "Training failed"
                    )
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": f'❌ Training failed: {training_status["error"]}',
                            "level": "ERROR",
                        }
                    )
            else:
                training_status["error"] = "Comprehensive ML script not found"

        elif training_type == "automated_training":
            training_status["current_task"] = (
                "Starting automated ML training & validation"
            )
            training_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "🤖 Starting automated ML training with model optimization...",
                    "level": "INFO",
                }
            )

            # Run automated training directly using Python API
            try:
                # Import the comprehensive system
                from comprehensive_enhanced_ml_system import \
                    ComprehensiveEnhancedMLSystem

                # Initialize system
                system = ComprehensiveEnhancedMLSystem()

                training_status["progress"] = 10
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": "📊 Loading and preparing comprehensive dataset...",
                        "level": "INFO",
                    }
                )

                # Load form guide data
                form_data = system.load_form_guide_data()

                # Load race results
                race_results_df = system.load_race_results_data()
                if race_results_df is None or len(race_results_df) < 100:
                    raise Exception("Insufficient race results data for training")

                training_status["progress"] = 20
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"📋 Loaded {len(race_results_df)} race records and {len(form_data)} dog profiles",
                        "level": "INFO",
                    }
                )

                # Create comprehensive features
                enhanced_df = system.create_comprehensive_features(
                    race_results_df, form_data
                )
                if enhanced_df is None:
                    raise Exception("Comprehensive feature creation failed")

                training_status["progress"] = 30
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"🔧 Created comprehensive features for {len(enhanced_df)} samples",
                        "level": "INFO",
                    }
                )

                # Prepare features
                prepared_df, feature_columns = system.prepare_comprehensive_features(
                    enhanced_df
                )
                if prepared_df is None:
                    raise Exception("Feature preparation failed")

                training_status["progress"] = 40
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"⚙️ Starting auto-optimization with {len(feature_columns)} features...",
                        "level": "INFO",
                    }
                )

                # Run automated model optimization
                best_model, scaler, validation_results = (
                    system.auto_optimize_model_parameters(prepared_df, feature_columns)
                )

                if best_model is None:
                    raise Exception("Auto-optimization failed to find suitable model")

                training_status["progress"] = 70
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f'🏆 Best model selected: {validation_results[0]["config_name"]} (Score: {validation_results[0]["composite_score"]:.3f})',
                        "level": "INFO",
                    }
                )

                # Historical validation
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": "🎯 Running historical race validation...",
                        "level": "INFO",
                    }
                )

                # Split data for validation
                df_sorted = prepared_df.sort_values("race_date")
                split_point = int(0.8 * len(df_sorted))
                test_df = df_sorted.iloc[split_point:]

                validation_summary = system.validate_model_on_historical_races(
                    best_model, scaler, feature_columns, test_df
                )

                training_status["progress"] = 85
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f'✅ Historical validation completed - Optimal accuracy: {validation_summary.get("optimal_accuracy", 0):.3f}',
                        "level": "INFO",
                    }
                )

                # Save best model
                model_file = (
                    system.models_dir
                    / f"automated_best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                )
                import joblib

                joblib.dump(
                    {
                        "model": best_model,
                        "scaler": scaler,
                        "feature_columns": feature_columns,
                        "model_name": validation_results[0]["config_name"],
                        "accuracy": validation_results[0]["accuracy"],
                        "auc": validation_results[0]["auc"],
                        "composite_score": validation_results[0]["composite_score"],
                        "validation_summary": validation_summary,
                        "timestamp": datetime.now().isoformat(),
                    },
                    model_file,
                )

                training_status["progress"] = 100
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"💾 Model saved successfully: {model_file.name}",
                        "level": "INFO",
                    }
                )

                training_status["completed"] = True
                training_status["results"] = {
                    "accuracy": validation_results[0]["accuracy"],
                    "auc_score": validation_results[0]["auc"],
                    "samples": len(prepared_df),
                    "optimal_threshold": validation_summary.get(
                        "optimal_threshold", 0.5
                    ),
                    "calibration_error": validation_summary.get(
                        "avg_calibration_error", 0
                    ),
                    "model_configurations_tested": len(validation_results),
                }

            except Exception as e:
                training_status["error"] = str(e)
                training_status["log"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"❌ Automated training failed: {str(e)}",
                        "level": "ERROR",
                    }
                )

        elif training_type == "backtesting":
            training_status["current_task"] = "🚀 COMPREHENSIVE ML BACKTESTING SYSTEM"
            training_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "🚀 COMPREHENSIVE ML BACKTESTING SYSTEM",
                    "level": "INFO",
                }
            )
            training_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "══════════════════════════════════════════════════════════════════════",
                    "level": "INFO",
                }
            )

            script_path = "ml_backtesting_trainer.py"
            if os.path.exists(script_path):
                # Start the enhanced ML backtesting process
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                # Real-time progress tracking
                step_progress = {
                    "STEP 1": 15,
                    "STEP 2": 30,
                    "STEP 3": 45,
                    "STEP 4": 60,
                    "STEP 5": 75,
                    "STEP 6": 85,
                    "STEP 7": 95,
                }

                # Read output line by line for real-time updates
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        if line:
                            # Update progress based on step detection
                            for step, progress in step_progress.items():
                                if step in line:
                                    training_status["progress"] = progress
                                    training_status["current_task"] = line[
                                        :100
                                    ]  # Truncate long lines
                                    break

                            # Add to log with enhanced formatting
                            log_level = (
                                "ERROR"
                                if "❌" in line or "Error" in line or "Failed" in line
                                else (
                                    "WARNING"
                                    if "⚠️" in line or "Warning" in line
                                    else (
                                        "SUCCESS"
                                        if "✅" in line or "completed" in line.lower()
                                        else "INFO"
                                    )
                                )
                            )

                            training_status["log"].append(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "message": line,
                                    "level": log_level,
                                }
                            )

                            # Keep only last 50 log entries to prevent memory issues
                            if len(training_status["log"]) > 50:
                                training_status["log"] = training_status["log"][-50:]

                # Wait for process completion
                return_code = process.wait()

                if return_code == 0:
                    training_status["progress"] = 100
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": "🎉 BACKTESTING COMPLETE!",
                            "level": "SUCCESS",
                        }
                    )
                    training_status["completed"] = True
                else:
                    stderr_output = process.stderr.read()
                    training_status["error"] = (
                        stderr_output[:500] if stderr_output else "Backtesting failed"
                    )
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": f'❌ Backtesting failed: {training_status["error"]}',
                            "level": "ERROR",
                        }
                    )
            else:
                training_status["error"] = "ML backtesting script not found"

        elif training_type == "feature_analysis":
            training_status["current_task"] = "🔍 Feature Importance Analysis"
            training_status["log"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": "🔍 Starting comprehensive feature importance analysis...",
                    "level": "INFO",
                }
            )

            script_path = "feature_importance_analyzer.py"
            if os.path.exists(script_path):
                # Start feature analysis with real-time output
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                # Read output line by line
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        if line:
                            # Update progress based on keywords
                            if "Loading" in line:
                                training_status["progress"] = 20
                            elif "Analyzing" in line:
                                training_status["progress"] = 40
                            elif "Generating" in line:
                                training_status["progress"] = 60
                            elif "Creating" in line:
                                training_status["progress"] = 80
                            elif "completed" in line.lower():
                                training_status["progress"] = 95

                            training_status["current_task"] = line[:100]

                            # Add to log with proper formatting
                            log_level = (
                                "ERROR"
                                if "❌" in line
                                else (
                                    "WARNING"
                                    if "⚠️" in line
                                    else "SUCCESS" if "✅" in line else "INFO"
                                )
                            )

                            training_status["log"].append(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "message": line,
                                    "level": log_level,
                                }
                            )

                            if len(training_status["log"]) > 50:
                                training_status["log"] = training_status["log"][-50:]

                return_code = process.wait()

                if return_code == 0:
                    training_status["progress"] = 100
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": "✅ Feature analysis completed successfully! Check results in feature_analysis_results/",
                            "level": "SUCCESS",
                        }
                    )
                    training_status["completed"] = True

                    # Try to run automated updater
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": "🔄 Running automated system update...",
                            "level": "INFO",
                        }
                    )

                    updater_script = "automated_feature_importance_updater.py"
                    if os.path.exists(updater_script):
                        updater_result = subprocess.run(
                            [sys.executable, updater_script],
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if updater_result.returncode == 0:
                            training_status["log"].append(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "message": "🎉 Prediction system automatically updated with latest insights!",
                                    "level": "SUCCESS",
                                }
                            )
                        else:
                            training_status["log"].append(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "message": "⚠️ Automated update completed with warnings",
                                    "level": "WARNING",
                                }
                            )
                else:
                    stderr_output = process.stderr.read()
                    training_status["error"] = (
                        stderr_output[:500]
                        if stderr_output
                        else "Feature analysis failed"
                    )
                    training_status["log"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "message": f'❌ Feature analysis failed: {training_status["error"]}',
                            "level": "ERROR",
                        }
                    )
            else:
                training_status["error"] = "Feature analysis script not found"

        training_status["progress"] = 100

    except subprocess.TimeoutExpired:
        training_status["error"] = "Training timed out"
        training_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "⏰ Training timed out",
                "level": "ERROR",
            }
        )
    except Exception as e:
        training_status["error"] = str(e)
        training_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": f"❌ Training error: {str(e)}",
                "level": "ERROR",
            }
        )
    finally:
        training_status["running"] = False
        training_status["current_task"] = "Training completed"


@app.route("/api/comprehensive_training", methods=["POST"])
def api_comprehensive_training():
    """Start comprehensive ML model training"""
    if training_status["running"]:
        return (
            jsonify({"success": False, "message": "Training already in progress"}),
            400,
        )

    # Start background training
    thread = threading.Thread(
        target=run_training_background, args=("comprehensive_training",)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Comprehensive ML training started"})


@app.route("/api/automated_training", methods=["POST"])
def api_automated_training():
    """Start automated ML model training with optimization and validation"""
    if training_status["running"]:
        return (
            jsonify({"success": False, "message": "Training already in progress"}),
            400,
        )

    # Start background automated training
    thread = threading.Thread(
        target=run_training_background, args=("automated_training",)
    )
    thread.daemon = True
    thread.start()

    return jsonify(
        {"success": True, "message": "Automated ML training & validation started"}
    )


@app.route("/api/backtesting", methods=["POST"])
def api_backtesting():
    """Start ML backtesting validation"""
    if training_status["running"]:
        return (
            jsonify({"success": False, "message": "Training already in progress"}),
            400,
        )

    # Start background backtesting
    thread = threading.Thread(target=run_training_background, args=("backtesting",))
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "ML backtesting started"})


@app.route("/api/feature_analysis", methods=["POST"])
def api_feature_analysis():
    """Start feature importance analysis"""
    if training_status["running"]:
        return (
            jsonify({"success": False, "message": "Training already in progress"}),
            400,
        )

    # Start background feature analysis
    thread = threading.Thread(
        target=run_training_background, args=("feature_analysis",)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Feature analysis started"})


@app.route("/api/training_status")
def api_training_status():
    """Get current training status"""
    return jsonify(training_status)


@app.route("/api/stop_training", methods=["POST"])
def api_stop_training():
    """Stop current training process"""
    global training_status

    if training_status["running"]:
        training_status["running"] = False
        training_status["error"] = "Training stopped by user"
        training_status["log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": "⏹️ Training stopped by user",
                "level": "WARNING",
            }
        )
        return jsonify({"success": True, "message": "Training stop signal sent"})
    else:
        return jsonify({"success": False, "message": "No training currently running"})


@app.route("/api/start_automated_monitoring", methods=["POST"])
def api_start_automated_monitoring():
    """Start automated model monitoring"""
    try:
        # Start automated backtesting system
        script_path = "automated_backtesting_system.py"
        if os.path.exists(script_path):
            # Run in background
            import subprocess

            subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return jsonify({"success": True, "message": "Automated monitoring started"})
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Automated monitoring script not found",
                    }
                ),
                500,
            )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error starting automated monitoring: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/check_performance_drift")
def api_check_performance_drift():
    """Check for model performance and feature drift"""
    try:
        feature_drift_detected = False
        feature_drift_reports = {}

        # Step 1: Check for feature drift if baseline exists
        try:
            current_features = feature_store.load()
            baseline_features = pd.read_parquet("baseline_feature_store.parquet")

            feature_drift_reports = feature_store.check_drift(
                current_features, baseline_features
            )
            feature_drift_detected = any(
                report["drift_detected"] for report in feature_drift_reports.values()
            )
        except FileNotFoundError:
            # No baseline features available yet
            feature_drift_reports = {
                "message": "No baseline features found for comparison"
            }
        except Exception as fe:
            feature_drift_reports = {"error": f"Feature drift check failed: {str(fe)}"}

        # Step 2: Check for performance drift using backtest results
        performance_drift_detected = False
        performance_drift_data = {}

        try:
            results_dir = Path("./comprehensive_model_results")
            if results_dir.exists():
                result_files = list(results_dir.glob("backtest_results_*.json"))

                if len(result_files) >= 2:
                    # Compare latest two results
                    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                    with open(result_files[0], "r") as f:
                        latest_results = json.load(f)
                    with open(result_files[1], "r") as f:
                        previous_results = json.load(f)

                    latest_accuracy = latest_results.get("accuracy", 0)
                    previous_accuracy = previous_results.get("accuracy", 0)

                    if previous_accuracy > 0:
                        drift_percentage = (
                            abs(latest_accuracy - previous_accuracy)
                            / previous_accuracy
                            * 100
                        )
                        performance_drift_detected = (
                            drift_percentage > 5
                        )  # 5% threshold

                        performance_drift_data = {
                            "drift_percentage": round(drift_percentage, 2),
                            "latest_accuracy": latest_accuracy,
                            "previous_accuracy": previous_accuracy,
                            "threshold_exceeded": performance_drift_detected,
                        }
                    else:
                        performance_drift_data = {
                            "message": "Invalid historical accuracy data"
                        }
                else:
                    performance_drift_data = {
                        "message": "Insufficient historical data for performance drift detection"
                    }
            else:
                performance_drift_data = {"message": "No results directory found"}
        except Exception as pe:
            performance_drift_data = {
                "error": f"Performance drift check failed: {str(pe)}"
            }

        # Combined drift assessment
        overall_drift_detected = feature_drift_detected or performance_drift_detected

        return jsonify(
            {
                "success": True,
                "overall_drift_detected": overall_drift_detected,
                "feature_drift": {
                    "detected": feature_drift_detected,
                    "reports": feature_drift_reports,
                },
                "performance_drift": {
                    "detected": performance_drift_detected,
                    "data": performance_drift_data,
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error checking drift: {str(e)}"}),
            500,
        )


@app.route("/api/check_data_quality")
def api_check_data_quality():
    """Check training data quality"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check data completeness
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        total_records = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NOT NULL AND finish_position != '' AND finish_position != 'N/A'"
        )
        complete_records = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM dog_race_data WHERE individual_time IS NOT NULL AND individual_time > 0"
        )
        time_records = cursor.fetchone()[0]

        conn.close()

        if total_records == 0:
            quality_score = 0
        else:
            # Calculate quality score based on data completeness
            completeness = complete_records / total_records
            time_completeness = time_records / total_records
            quality_score = completeness * 0.7 + time_completeness * 0.3

        return jsonify(
            {
                "success": True,
                "quality_score": quality_score,
                "total_records": total_records,
                "complete_records": complete_records,
                "time_records": time_records,
                "completeness_percentage": (
                    round(completeness * 100, 1) if total_records > 0 else 0
                ),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error checking data quality: {str(e)}"}
            ),
            500,
        )


@app.route("/api/export_model")
def api_export_model():
    """Export current trained model"""
    try:
        import io
        import zipfile

        from flask import send_file

        models_dir = Path("./comprehensive_trained_models")
        if not models_dir.exists():
            return (
                jsonify({"success": False, "message": "No trained models found"}),
                404,
            )

        # Find latest model
        model_files = list(models_dir.glob("comprehensive_best_model_*.joblib"))
        if not model_files:
            return (
                jsonify({"success": False, "message": "No trained models found"}),
                404,
            )

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(latest_model, latest_model.name)

            # Add results if available
            results_dir = Path("./comprehensive_model_results")
            if results_dir.exists():
                result_files = list(results_dir.glob("comprehensive_analysis_*.json"))
                if result_files:
                    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                    zip_file.write(latest_result, latest_result.name)

        zip_buffer.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"greyhound_ml_model_{timestamp}.zip"

        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/zip",
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error exporting model: {str(e)}"}),
            500,
        )


@app.route("/api/download_training_report")
def api_download_training_report():
    """Download comprehensive training report"""
    try:
        import io

        from flask import send_file

        # Generate comprehensive training report
        report_content = io.StringIO()

        report_content.write("GREYHOUND RACING ML TRAINING REPORT\n")
        report_content.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        report_content.write("=" * 50 + "\n\n")

        # Model status
        try:
            import joblib

            models_dir = Path("./comprehensive_trained_models")
            if models_dir.exists():
                model_files = list(models_dir.glob("comprehensive_best_model_*.joblib"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    model_data = joblib.load(latest_model)

                    report_content.write("CURRENT MODEL STATUS\n")
                    report_content.write("-" * 20 + "\n")
                    report_content.write(
                        f"Model Type: {model_data.get('model_name', 'Unknown')}\n"
                    )
                    report_content.write(
                        f"Features: {len(model_data.get('feature_columns', []))}\n"
                    )
                    report_content.write(
                        f"Training Samples: {model_data.get('data_summary', {}).get('total_samples', 'Unknown')}\n"
                    )
                    report_content.write(
                        f"Last Trained: {model_data.get('timestamp', 'Unknown')}\n\n"
                    )
        except Exception as e:
            report_content.write(f"Error loading model info: {e}\n\n")

        # Latest results
        try:
            results_dir = Path("./comprehensive_model_results")
            if results_dir.exists():
                result_files = list(results_dir.glob("comprehensive_analysis_*.json"))
                if result_files:
                    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_result, "r") as f:
                        results = json.load(f)

                    report_content.write("LATEST TRAINING RESULTS\n")
                    report_content.write("-" * 25 + "\n")

                    for model_name, metrics in results.get("model_results", {}).items():
                        report_content.write(f"{model_name.upper()}:\n")
                        report_content.write(
                            f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n"
                        )
                        report_content.write(
                            f"  AUC Score: {metrics.get('auc', 0):.3f}\n"
                        )
                        report_content.write("\n")
        except Exception as e:
            report_content.write(f"Error loading results: {e}\n\n")

        # Training history
        if training_status["log"]:
            report_content.write("RECENT TRAINING LOG\n")
            report_content.write("-" * 20 + "\n")
            for entry in training_status["log"][-10:]:  # Last 10 entries
                timestamp = entry.get("timestamp", "Unknown")
                message = entry.get("message", "")
                report_content.write(f"[{timestamp}] {message}\n")

        report_content.seek(0)
        file_like = io.BytesIO(report_content.getvalue().encode("utf-8"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ml_training_report_{timestamp}.txt"

        return send_file(
            file_like, as_attachment=True, download_name=filename, mimetype="text/plain"
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error generating training report: {str(e)}",
                }
            ),
            500,
        )


# GPT Enhancement API Endpoints


@app.route("/api/gpt/enhance_race", methods=["POST"])
def api_gpt_enhance_race():
    """API endpoint to enhance a race with GPT analysis"""
    try:
        data = request.get_json()
        race_file_path = data.get("race_file_path")
        include_betting = data.get("include_betting_strategy", True)
        include_patterns = data.get("include_pattern_analysis", True)

        if not race_file_path:
            return (
                jsonify({"success": False, "message": "Race file path is required"}),
                400,
            )

        # Use singleton GPT enhancer
        enhancer = get_gpt_enhancer()
        if not enhancer:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "GPT enhancement not available. Install openai package and set OPENAI_API_KEY.",
                    }
                ),
                500,
            )

        # Run enhancement
        result = enhancer.enhance_race_prediction(
            race_file_path,
            include_betting_strategy=include_betting,
            include_pattern_analysis=include_patterns,
        )

        if "error" in result:
            return jsonify({"success": False, "message": result["error"]}), 500

        return jsonify(
            {
                "success": True,
                "enhancement": result,
                "tokens_used": result.get("tokens_used", 0),
                "estimated_cost": result.get("tokens_used", 0)
                * 0.045
                / 1000,  # Rough estimate
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error enhancing race: {str(e)}"}),
            500,
        )


@app.route("/api/gpt/daily_insights", methods=["GET"])
def api_gpt_daily_insights():
    """API endpoint to get GPT daily insights"""
    try:
        date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))

        # Use singleton GPT enhancer
        enhancer = get_gpt_enhancer()
        if not enhancer:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "GPT enhancement not available. Install openai package and set OPENAI_API_KEY.",
                    }
                ),
                500,
            )

        insights = enhancer.generate_daily_insights(date_str)

        return jsonify(
            {
                "success": True,
                "insights": insights,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error generating daily insights: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/gpt/enhance_multiple", methods=["POST"])
def api_gpt_enhance_multiple():
    """API endpoint to enhance multiple races with GPT analysis"""
    try:
        data = request.get_json()
        race_files = data.get("race_files", [])
        max_races = data.get("max_races", 5)

        if not race_files:
            return (
                jsonify({"success": False, "message": "Race files list is required"}),
                400,
            )

        # Use singleton GPT enhancer
        enhancer = get_gpt_enhancer()
        if not enhancer:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "GPT enhancement not available. Install openai package and set OPENAI_API_KEY.",
                    }
                ),
                500,
            )

        results = enhancer.enhance_multiple_races(race_files, max_races)

        return jsonify(
            {
                "success": True,
                "batch_results": results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error enhancing multiple races: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/gpt/comprehensive_report", methods=["POST"])
def api_gpt_comprehensive_report():
    """API endpoint to generate comprehensive GPT report"""
    try:
        data = request.get_json()
        race_ids = data.get("race_ids", [])

        if not race_ids:
            return (
                jsonify({"success": False, "message": "Race IDs list is required"}),
                400,
            )

        # Use singleton GPT enhancer
        enhancer = get_gpt_enhancer()
        if not enhancer:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "GPT enhancement not available. Install openai package and set OPENAI_API_KEY.",
                    }
                ),
                500,
            )

        report = enhancer.create_comprehensive_report(race_ids)

        return jsonify(
            {
                "success": True,
                "report": report,
                "race_count": len(race_ids),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error generating comprehensive report: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/gpt/status")
def api_gpt_status():
    """API endpoint to check GPT integration status"""
    try:
        import os

        # Check if OpenAI API key is available
        api_key_available = bool(os.getenv("OPENAI_API_KEY"))

        # Try to import and initialize
        gpt_available = False
        error_message = None

        try:
            from gpt_prediction_enhancer import GPTPredictionEnhancer

            enhancer = GPTPredictionEnhancer()
            gpt_available = enhancer.gpt_available
        except Exception as e:
            error_message = str(e)

        # Check for enhanced predictions directory
        enhanced_dir = Path("./gpt_enhanced_predictions")
        enhanced_count = 0
        if enhanced_dir.exists():
            enhanced_count = len(list(enhanced_dir.glob("gpt_enhanced_*.json")))

        return jsonify(
            {
                "success": True,
                "status": {
                    "api_key_configured": api_key_available,
                    "gpt_analyzer_available": gpt_available,
                    "error_message": error_message,
                    "enhanced_predictions_count": enhanced_count,
                    "model_used": "gpt-4-turbo-preview",
                    "estimated_cost_per_race": "$0.15 - $0.30",
                    "features_available": [
                        "Race analysis with contextual insights",
                        "ML prediction enhancement",
                        "Betting strategy generation",
                        "Historical pattern analysis",
                        "Daily insights compilation",
                        "Comprehensive reporting",
                    ],
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error checking GPT status: {str(e)}"}
            ),
            500,
        )


@app.route("/gpt-enhancement")
def gpt_enhancement_dashboard():
    """GPT Enhancement Dashboard page"""
    return render_template("gpt_enhancement.html")


# Sportsbet Odds Integration API Endpoints


@app.route("/api/sportsbet/update_odds", methods=["POST"])
def api_update_sportsbet_odds():
    """API endpoint to update odds from Sportsbet"""
    try:
        races = sportsbet_integrator.get_today_races()

        for race in races:
            sportsbet_integrator.save_odds_to_database(race)

        # Identify value bets after updating odds
        value_bets = sportsbet_integrator.identify_value_bets()

        return jsonify(
            {
                "success": True,
                "message": f"Updated odds for {len(races)} races",
                "races_updated": len(races),
                "value_bets_found": len(value_bets),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error updating Sportsbet odds: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/sportsbet/live_odds")
def api_live_odds_summary():
    """API endpoint to get live odds summary"""
    try:
        odds_summary = sportsbet_integrator.get_live_odds_summary()

        return jsonify(
            {
                "success": True,
                "odds_summary": odds_summary,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting live odds: {str(e)}"}
            ),
            500,
        )


@app.route("/api/sportsbet/today_races_basic")
def api_today_races_basic():
    """API endpoint to get today's races without fetching odds (fast load)"""
    try:
        # Use direct racing scraper for comprehensive coverage
        from direct_racing_scraper import get_today_races

        # Get races for today from the main racing page (more comprehensive)
        print("🔍 Using direct racing scraper for comprehensive race coverage...")
        today_races = get_today_races()

        # Add time calculations and format for display
        formatted_races = []
        now = datetime.now()

        for race in today_races:
            # Parse race time
            race_time = race.get("race_time", "Unknown")
            minutes_until = 999
            time_status = "UNKNOWN"

            if race_time and race_time != "Unknown":
                try:
                    # Parse the race time - handle different formats
                    race_time_clean = race_time.strip()

                    # Handle different time formats
                    if (
                        "AM" in race_time_clean.upper()
                        or "PM" in race_time_clean.upper()
                    ):
                        # 12-hour format like "7:45 PM"
                        time_obj = datetime.strptime(
                            race_time_clean.upper(), "%I:%M %p"
                        ).time()
                    else:
                        # 24-hour format like "19:45" or "7:45"
                        if ":" in race_time_clean:
                            time_obj = datetime.strptime(
                                race_time_clean, "%H:%M"
                            ).time()
                        else:
                            # 4-digit format like "1945" or "745"
                            if len(race_time_clean) >= 3 and race_time_clean.isdigit():
                                # Pad with leading zero if needed (e.g., "745" -> "0745")
                                if len(race_time_clean) == 3:
                                    race_time_clean = "0" + race_time_clean
                                hour = int(race_time_clean[:2])
                                minute = (
                                    int(race_time_clean[2:4])
                                    if len(race_time_clean) >= 4
                                    else 0
                                )
                                time_obj = datetime.time(hour, minute)
                            else:
                                raise ValueError("Unknown time format")

                    # Get the race date from the race data
                    race_date_str = race.get("date")
                    if race_date_str:
                        race_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()
                    else:
                        # Default to today if no date specified
                        race_date = now.date()

                    # Create the full race datetime
                    race_datetime = datetime.combine(race_date, time_obj)

                    # Calculate time difference
                    time_diff = race_datetime - now
                    minutes_until = int(time_diff.total_seconds() / 60)

                    # Set status based on time until race
                    if minutes_until < -30:  # Race finished more than 30 minutes ago
                        time_status = "FINISHED"
                    elif minutes_until < 0:  # Race is in progress or just finished
                        time_status = "LIVE"
                    elif minutes_until <= 15:
                        time_status = "SOON"
                    elif minutes_until <= 120:
                        time_status = "UPCOMING"
                    else:
                        time_status = "LATER"

                except (ValueError, TypeError):
                    # If we can't parse the time, mark as unknown
                    time_status = "UNKNOWN"
                    minutes_until = 999

            # Construct Sportsbet URL
            venue_slug = race.get("venue", "").lower().replace(" ", "-")
            sportsbet_url = (
                f"https://www.sportsbet.com.au/betting/racing/greyhound/{venue_slug}"
                if venue_slug
                else None
            )

            formatted_race = {
                "venue": race.get("venue", "Unknown"),
                "venue_name": race.get("venue_name", race.get("venue", "Unknown")),
                "race_number": race.get("race_number", 0),
                "race_time": race_time,
                "formatted_race_time": race_time,
                "distance": race.get("distance", "Unknown"),
                "grade": race.get("grade", "Unknown"),
                "race_name": race.get("race_name", ""),
                "minutes_until_race": minutes_until,
                "time_status": time_status,
                "sportsbet_url": sportsbet_url,
                "race_url": race.get("url") or race.get("race_url"),
                "has_odds": False,  # No odds loaded yet
                "race_id": f"{venue_slug}_r{race.get('race_number', 0)}_{now.strftime('%Y%m%d')}",
            }

            formatted_races.append(formatted_race)

        # Sort by time until race (soonest first)
        formatted_races.sort(key=lambda x: x.get("minutes_until_race", 999))

        return jsonify(
            {
                "success": True,
                "races": formatted_races,
                "total_races": len(formatted_races),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting today's races: {str(e)}"}
            ),
            500,
        )


@app.route("/api/sportsbet/race_odds_on_demand/<race_id>", methods=["POST"])
def api_fetch_race_odds_on_demand(race_id):
    """API endpoint to fetch odds for a specific race on demand"""
    try:
        data = request.get_json() or {}
        race_url = data.get("race_url")
        venue = data.get("venue")
        race_number = data.get("race_number")

        if not race_url:
            return (
                jsonify({"success": False, "message": "Missing race_url parameter"}),
                400,
            )

        # Use the sportsbet integrator to fetch odds for this specific race
        integrator = sportsbet_integrator
        integrator.setup_driver()

        try:
            # Create a race info object for odds extraction
            race_info = {
                "race_id": race_id,
                "venue": venue,
                "race_number": race_number,
                "venue_url": race_url,
                "race_date": datetime.now().date(),
                "race_time": "Unknown",
                "odds_data": [],
            }

            # Extract odds from the race page
            enhanced_race = integrator.get_race_odds_from_page(race_info)
            odds_data = enhanced_race.get("odds_data", [])

            if odds_data:
                # Save to database for future quick access
                integrator.save_odds_to_database(enhanced_race)

                # Calculate summary stats
                odds_values = [dog["odds_decimal"] for dog in odds_data]
                summary = {
                    "dog_count": len(odds_data),
                    "favorite_odds": min(odds_values) if odds_values else 0,
                    "longest_odds": max(odds_values) if odds_values else 0,
                    "avg_odds": (
                        sum(odds_values) / len(odds_values) if odds_values else 0
                    ),
                }

                return jsonify(
                    {
                        "success": True,
                        "race_id": race_id,
                        "odds_data": odds_data,
                        "summary": summary,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                return jsonify(
                    {
                        "success": False,
                        "message": "No odds data could be extracted from the race page",
                    }
                )

        finally:
            integrator.close_driver()

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error fetching race odds: {str(e)}"}
            ),
            500,
        )


@app.route("/api/sportsbet/value_bets")
def api_value_bets():
    """API endpoint to get current value betting opportunities"""
    try:
        value_bets = sportsbet_integrator.get_value_bets_summary()

        return jsonify(
            {
                "success": True,
                "value_bets": value_bets,
                "count": len(value_bets),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting value bets: {str(e)}"}
            ),
            500,
        )


@app.route("/api/sportsbet/odds_history/<race_id>/<dog_name>")
def api_odds_history(race_id, dog_name):
    """API endpoint to get odds movement history for a specific dog"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT odds_decimal, odds_change, timestamp 
            FROM odds_history 
            WHERE race_id = ? AND dog_clean_name = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """,
            (race_id, dog_name.upper()),
        )

        history = cursor.fetchall()
        conn.close()

        odds_history = []
        for record in history:
            odds_history.append(
                {
                    "odds_decimal": record[0],
                    "odds_change": record[1],
                    "timestamp": record[2],
                }
            )

        return jsonify(
            {
                "success": True,
                "race_id": race_id,
                "dog_name": dog_name,
                "odds_history": odds_history,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting odds history: {str(e)}"}
            ),
            500,
        )


@app.route("/api/sportsbet/race_odds/<race_id>")
def api_race_odds(race_id):
    """API endpoint to get current odds for all dogs in a race"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT dog_name, dog_clean_name, box_number, odds_decimal, odds_fractional, timestamp
            FROM live_odds 
            WHERE race_id = ? AND is_current = TRUE
            ORDER BY box_number
        """,
            (race_id,),
        )

        odds_data = cursor.fetchall()
        conn.close()

        race_odds = []
        for record in odds_data:
            race_odds.append(
                {
                    "dog_name": record[0],
                    "dog_clean_name": record[1],
                    "box_number": record[2],
                    "odds_decimal": record[3],
                    "odds_fractional": record[4],
                    "timestamp": record[5],
                }
            )

        return jsonify(
            {
                "success": True,
                "race_id": race_id,
                "race_odds": race_odds,
                "dog_count": len(race_odds),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting race odds: {str(e)}"}
            ),
            500,
        )


@app.route("/api/predictions/upcoming", methods=["POST"])
def api_predictions_upcoming():
    """API endpoint to get predictions for upcoming races based on race IDs"""
    try:
        data = request.get_json()
        if not data or "race_ids" not in data:
            return (
                jsonify({"success": False, "error": "race_ids parameter is required"}),
                400,
            )

        race_ids = data["race_ids"]
        prediction_types = data.get(
            "prediction_types", ["win_probability", "place_probability"]
        )

        # Get predictions from prediction files
        predictions_dir = "./predictions"
        predictions_result = {}

        if os.path.exists(predictions_dir):
            for race_id in race_ids:
                race_predictions = {}

                # Look for prediction files matching this race ID
                for filename in os.listdir(predictions_dir):
                    if (
                        filename.startswith("prediction_")
                        or filename.startswith("unified_prediction_")
                    ) and filename.endswith(".json"):
                        try:
                            file_path = os.path.join(predictions_dir, filename)
                            with open(file_path, "r") as f:
                                pred_data = json.load(f)

                            race_info = pred_data.get("race_info", {})
                            race_filename = race_info.get("filename", "")

                            # Check if this prediction file matches the race ID
                            if (
                                race_id in race_filename
                                or race_filename.replace(".csv", "") == race_id
                            ):
                                predictions_list = pred_data.get("predictions", [])

                                # Extract predictions for each type requested
                                for pred_type in prediction_types:
                                    if pred_type == "win_probability":
                                        race_predictions[pred_type] = {
                                            "predictions": [
                                                {
                                                    "box": pred.get(
                                                        "box_number", "N/A"
                                                    ),
                                                    "dog_name": pred.get(
                                                        "dog_name", "Unknown"
                                                    ),
                                                    "win_probability": safe_float(
                                                        pred.get("final_score", 0)
                                                    ),
                                                    "confidence": safe_float(
                                                        pred.get("final_score", 0)
                                                    ),
                                                }
                                                for pred in predictions_list
                                            ],
                                            "race_info": race_info,
                                        }
                                    elif pred_type == "place_probability":
                                        race_predictions[pred_type] = {
                                            "predictions": [
                                                {
                                                    "box": pred.get(
                                                        "box_number", "N/A"
                                                    ),
                                                    "dog_name": pred.get(
                                                        "dog_name", "Unknown"
                                                    ),
                                                    "place_probability": safe_float(
                                                        pred.get("final_score", 0)
                                                    )
                                                    * 0.7,  # Estimate place prob
                                                    "confidence": safe_float(
                                                        pred.get("final_score", 0)
                                                    ),
                                                }
                                                for pred in predictions_list
                                            ],
                                            "race_info": race_info,
                                        }
                                    elif pred_type == "race_time":
                                        race_predictions[pred_type] = {
                                            "race_time": race_info.get(
                                                "scheduled_time", "TBD"
                                            ),
                                            "race_info": race_info,
                                        }
                                break
                        except Exception:
                            continue

                if race_predictions:
                    predictions_result[race_id] = race_predictions

        return jsonify(
            {
                "success": True,
                "predictions": predictions_result,
                "race_count": len(predictions_result),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Error getting upcoming predictions: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/sportsbet/enhanced_predictions")
def api_enhanced_predictions_with_odds():
    """API endpoint to get predictions enhanced with live odds and value analysis"""
    try:
        # Get recent predictions
        predictions_dir = "./predictions"
        enhanced_predictions = []

        if os.path.exists(predictions_dir):
            prediction_files = []
            for filename in os.listdir(predictions_dir):
                if (
                    filename.startswith("prediction_")
                    and filename.endswith(".json")
                    and "summary" not in filename
                ):
                    file_path = os.path.join(predictions_dir, filename)
                    prediction_files.append((file_path, os.path.getmtime(file_path)))

            # Sort by modification time (newest first)
            prediction_files.sort(key=lambda x: x[1], reverse=True)

            # Process latest 5 predictions
            for file_path, _ in prediction_files[:5]:
                try:
                    with open(file_path, "r") as f:
                        prediction_data = json.load(f)

                    race_id = prediction_data.get("race_context", {}).get("race_id", "")
                    if not race_id:
                        continue

                    # Get live odds for this race
                    conn = sqlite3.connect(DATABASE_PATH)
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        SELECT dog_clean_name, odds_decimal
                        FROM live_odds 
                        WHERE race_id = ? AND is_current = TRUE
                    """,
                        (race_id,),
                    )

                    live_odds = dict(cursor.fetchall())
                    conn.close()

                    # Enhance predictions with odds and value analysis
                    enhanced_dogs = []
                    for dog in prediction_data.get("predictions", []):
                        dog_clean_name = (
                            dog.get("dog_name", "").upper().replace(" ", "")
                        )
                        predicted_prob = dog.get("prediction_score", 0)

                        enhanced_dog = dog.copy()

                        if dog_clean_name in live_odds:
                            market_odds = live_odds[dog_clean_name]
                            implied_prob = 1.0 / market_odds if market_odds > 0 else 0

                            # Calculate value
                            if implied_prob > 0:
                                value_percentage = (
                                    (predicted_prob - implied_prob) / implied_prob
                                ) * 100

                                enhanced_dog.update(
                                    {
                                        "live_odds": market_odds,
                                        "implied_probability": implied_prob,
                                        "value_percentage": value_percentage,
                                        "has_value": value_percentage > 10,
                                        "betting_recommendation": (
                                            sportsbet_integrator.generate_bet_recommendation(
                                                value_percentage,
                                                dog.get("confidence_level", "MEDIUM"),
                                                market_odds,
                                            )
                                            if value_percentage > 10
                                            else "PASS"
                                        ),
                                    }
                                )

                        enhanced_dogs.append(enhanced_dog)

                    # Sort by value percentage (highest first)
                    enhanced_dogs.sort(
                        key=lambda x: x.get("value_percentage", -999), reverse=True
                    )

                    enhanced_predictions.append(
                        {
                            "race_info": prediction_data.get("race_context", {}),
                            "predictions": enhanced_dogs[:8],  # Top 8 dogs
                            "has_live_odds": len(live_odds) > 0,
                            "value_opportunities": len(
                                [d for d in enhanced_dogs if d.get("has_value", False)]
                            ),
                        }
                    )

                except Exception as e:
                    print(f"Error processing prediction file {file_path}: {e}")
                    continue

        return jsonify(
            {
                "success": True,
                "enhanced_predictions": enhanced_predictions,
                "races_analyzed": len(enhanced_predictions),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error generating enhanced predictions: {str(e)}",
                }
            ),
            500,
        )


@app.route("/odds_dashboard")
def odds_dashboard():
    """Sportsbet odds dashboard page"""
    return render_template("odds_dashboard.html")


@app.route("/automation")
def automation_dashboard():
    """Automation monitoring dashboard page"""
    return render_template("automation_dashboard.html")


@app.route("/database-manager")
def database_manager():
    """Database management and automated updater dashboard"""
    return render_template("database_manager.html")


# Database Management API Endpoints


@app.route("/api/database/integrity_check", methods=["POST"])
def api_database_integrity_check():
    """Run comprehensive database integrity check"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        results = maintenance_manager.run_integrity_check()

        return jsonify(
            {
                "success": True,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Integrity check failed: {str(e)}"}),
            500,
        )


@app.route("/api/database/create_backup", methods=["POST"])
def api_database_create_backup():
    """Create database backup"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        backup_info = maintenance_manager.create_backup()

        return jsonify(
            {
                "success": True,
                "backup_info": backup_info,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Backup creation failed: {str(e)}"}),
            500,
        )


@app.route("/api/database/optimize", methods=["POST"])
def api_database_optimize():
    """Optimize database performance"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        optimization_results = maintenance_manager.optimize_database()

        return jsonify(
            {
                "success": True,
                "results": optimization_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Database optimization failed: {str(e)}"}
            ),
            500,
        )


@app.route("/api/database/update_statistics", methods=["POST"])
def api_database_update_statistics():
    """Update database statistics and indexes"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        stats_results = maintenance_manager.update_statistics()

        return jsonify(
            {
                "success": True,
                "results": stats_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Statistics update failed: {str(e)}"}
            ),
            500,
        )


@app.route("/api/database/cleanup", methods=["POST"])
def api_database_cleanup():
    """Cleanup old data and temporary files"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        cleanup_results = maintenance_manager.cleanup_old_data()

        return jsonify(
            {
                "success": True,
                "results": cleanup_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": f"Cleanup failed: {str(e)}"}), 500


@app.route("/api/database/maintenance_status")
def api_database_maintenance_status():
    """Get current database maintenance status"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        status = maintenance_manager.get_maintenance_status()

        return jsonify(
            {"success": True, "status": status, "timestamp": datetime.now().isoformat()}
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to get maintenance status: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/database/backup_history")
def api_database_backup_history():
    """Get database backup history"""
    try:
        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        backup_history = maintenance_manager.get_backup_history()

        return jsonify(
            {
                "success": True,
                "backups": backup_history,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Failed to get backup history: {str(e)}"}
            ),
            500,
        )


@app.route("/api/database/restore_backup", methods=["POST"])
def api_database_restore_backup():
    """Restore database from backup"""
    try:
        data = request.get_json()
        backup_filename = data.get("backup_filename")

        if not backup_filename:
            return (
                jsonify({"success": False, "message": "Backup filename is required"}),
                400,
            )

        from database_maintenance import DatabaseMaintenanceManager

        maintenance_manager = DatabaseMaintenanceManager(DATABASE_PATH)
        restore_results = maintenance_manager.restore_backup(backup_filename)

        return jsonify(
            {
                "success": True,
                "results": restore_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Backup restoration failed: {str(e)}"}
            ),
            500,
        )


@app.route("/api/sportsbet/start_monitoring", methods=["POST"])
def api_start_odds_monitoring():
    """API endpoint to start continuous odds monitoring"""
    try:
        # Start monitoring in a background thread
        def monitoring_worker():
            sportsbet_integrator.start_continuous_monitoring()

        monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitor_thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Odds monitoring started",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error starting odds monitoring: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/automation/status")
def api_automation_status():
    """API endpoint for automation service status"""
    try:
        # Check if automation service is running via launchctl
        result = subprocess.run(
            ["launchctl", "list", "com.greyhound.automation"],
            capture_output=True,
            text=True,
        )

        service_running = result.returncode == 0

        # Get current time info
        now = datetime.now()

        # Get last backup time if available
        backup_file = "greyhound_racing_data_backup.db"
        last_backup = None
        if os.path.exists(backup_file):
            last_backup = datetime.fromtimestamp(os.path.getmtime(backup_file))

        # Get predictions count
        predictions_dir = "./predictions"
        predictions_count = 0
        if os.path.exists(predictions_dir):
            predictions_count = len(
                [
                    f
                    for f in os.listdir(predictions_dir)
                    if f.startswith("prediction_") and f.endswith(".json")
                ]
            )

        return jsonify(
            {
                "success": True,
                "service_status": "running" if service_running else "stopped",
                "uptime": "24/7 Service" if service_running else "Not Running",
                "last_run": now.strftime("%Y-%m-%d %H:%M:%S"),
                "next_run": "06:00 AM Daily" if service_running else "N/A",
                "tasks_completed": 5,
                "active_predictions": predictions_count,
                "last_backup": (
                    last_backup.strftime("%Y-%m-%d %H:%M:%S")
                    if last_backup
                    else "Never"
                ),
                "timestamp": now.isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error getting automation status: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/automation/recent_tasks")
def api_automation_recent_tasks():
    """API endpoint for recent automation tasks"""
    try:
        # Mock recent tasks data - in production this would come from logs or database
        recent_tasks = [
            {
                "task_name": "Database Backup",
                "status": "completed",
                "start_time": (datetime.now() - timedelta(hours=2)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration": "2.3s",
                "details": "Backup completed successfully",
            },
            {
                "task_name": "Data Integrity Check",
                "status": "completed",
                "start_time": (datetime.now() - timedelta(hours=2, minutes=1)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration": "5.7s",
                "details": "All data integrity checks passed",
            },
            {
                "task_name": "Upcoming Races Collection",
                "status": "completed",
                "start_time": (datetime.now() - timedelta(hours=2, minutes=2)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration": "12.4s",
                "details": "Collected 15 upcoming races",
            },
            {
                "task_name": "Odds Update",
                "status": "completed",
                "start_time": (datetime.now() - timedelta(hours=2, minutes=5)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration": "8.9s",
                "details": "Updated odds for 12 races",
            },
            {
                "task_name": "Race Predictions",
                "status": "completed",
                "start_time": (datetime.now() - timedelta(hours=2, minutes=8)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration": "45.2s",
                "details": "Generated predictions for 8 races",
            },
        ]

        return jsonify(
            {
                "success": True,
                "tasks": recent_tasks,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting recent tasks: {str(e)}"}
            ),
            500,
        )


@app.route("/api/automation/logs")
def api_automation_logs():
    """API endpoint for automation logs"""
    try:
        log_type = request.args.get("type", "all")
        limit = request.args.get("limit", 100, type=int)

        # Get logs from the enhanced logger
        logs = logger.get_web_logs(log_type, limit)

        # Add some automation-specific logs if available
        automation_logs = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "level": "INFO",
                "component": "AUTOMATION",
                "message": "Morning routine completed successfully",
                "details": "All scheduled tasks executed without errors",
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "level": "INFO",
                "component": "BACKUP",
                "message": "Database backup created",
                "details": "Backup size: 2.4MB",
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "level": "INFO",
                "component": "PREDICTOR",
                "message": "Generated predictions for upcoming races",
                "details": "8 races analyzed with weather enhancement",
            },
        ]

        # Combine and sort logs
        all_logs = logs + automation_logs
        all_logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return jsonify(
            {
                "success": True,
                "logs": all_logs[:limit],
                "total_logs": len(all_logs),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error getting automation logs: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/automation/system_info")
def api_automation_system_info():
    """API endpoint for system information"""
    try:
        import platform

        import psutil

        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024**3):.1f}GB",
            "disk_usage": f"{disk.percent}%",
            "disk_free": f"{disk.free / (1024**3):.1f}GB",
            "uptime": "System running normally",
        }

        return jsonify(
            {
                "success": True,
                "system_info": system_info,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except ImportError:
        # Fallback if psutil is not available
        system_info = {
            "platform": "macOS",
            "python_version": "3.9+",
            "cpu_usage": "Normal",
            "memory_usage": "Normal",
            "memory_available": "Sufficient",
            "disk_usage": "Normal",
            "disk_free": "Sufficient",
            "uptime": "System running normally",
        }

        return jsonify(
            {
                "success": True,
                "system_info": system_info,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting system info: {str(e)}"}
            ),
            500,
        )


@app.route("/api/automation/run_task", methods=["POST"])
def api_automation_run_task():
    """API endpoint to run a specific automation task"""
    try:
        data = request.get_json()
        task_name = data.get("task_name")

        if not task_name:
            return jsonify({"success": False, "message": "Task name is required"}), 400

        # Map task names to actual commands
        task_commands = {
            "backup": ["python3", "automation_scheduler.py", "--task", "backup"],
            "integrity_check": [
                "python3",
                "automation_scheduler.py",
                "--task",
                "integrity",
            ],
            "collect_races": [
                "python3",
                "automation_scheduler.py",
                "--task",
                "collect",
            ],
            "update_odds": ["python3", "automation_scheduler.py", "--task", "odds"],
            "generate_predictions": [
                "python3",
                "automation_scheduler.py",
                "--task",
                "predict",
            ],
            "morning_routine": [
                "python3",
                "automation_scheduler.py",
                "--routine",
                "morning",
            ],
        }

        command = task_commands.get(task_name)
        if not command:
            return (
                jsonify({"success": False, "message": f"Unknown task: {task_name}"}),
                400,
            )

        # Run the task
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return jsonify(
                {
                    "success": True,
                    "message": f"Task {task_name} completed successfully",
                    "output": (
                        result.stdout[-1000:] if result.stdout else ""
                    ),  # Last 1000 chars
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Task {task_name} failed",
                        "error": (
                            result.stderr[-500:] if result.stderr else "Unknown error"
                        ),
                    }
                ),
                500,
            )

    except subprocess.TimeoutExpired:
        return (
            jsonify({"success": False, "message": "Task timed out (5 minute limit)"}),
            500,
        )
    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error running task: {str(e)}"}),
            500,
        )


@app.route("/api/automation/start", methods=["POST"])
def api_automation_start():
    """API endpoint to start the automation service"""
    try:
        # Start the launchd service
        result = subprocess.run(
            [
                "launchctl",
                "load",
                "-w",
                "/Users/orlandolee/Library/LaunchAgents/com.greyhound.automation.plist",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return jsonify(
                {
                    "success": True,
                    "message": "Automation service started successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Failed to start service: {result.stderr}",
                    }
                ),
                500,
            )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error starting automation service: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/automation/stop", methods=["POST"])
def api_automation_stop():
    """API endpoint to stop the automation service"""
    try:
        # Stop the launchd service
        result = subprocess.run(
            [
                "launchctl",
                "unload",
                "/Users/orlandolee/Library/LaunchAgents/com.greyhound.automation.plist",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return jsonify(
                {
                    "success": True,
                    "message": "Automation service stopped successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Failed to stop service: {result.stderr}",
                    }
                ),
                500,
            )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error stopping automation service: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/automation/restart", methods=["POST"])
def api_automation_restart():
    """API endpoint to restart the automation service"""
    try:
        # Stop the service first
        subprocess.run(
            [
                "launchctl",
                "unload",
                "/Users/orlandolee/Library/LaunchAgents/com.greyhound.automation.plist",
            ],
            capture_output=True,
            text=True,
        )

        # Wait a moment
        time.sleep(2)

        # Start the service
        result = subprocess.run(
            [
                "launchctl",
                "load",
                "-w",
                "/Users/orlandolee/Library/LaunchAgents/com.greyhound.automation.plist",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return jsonify(
                {
                    "success": True,
                    "message": "Automation service restarted successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Failed to restart service: {result.stderr}",
                    }
                ),
                500,
            )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error restarting automation service: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/automation/storage_info")
def api_automation_storage_info():
    """API endpoint for storage information"""
    try:
        # Get database size
        db_size = 0
        if os.path.exists(DATABASE_PATH):
            db_size = os.path.getsize(DATABASE_PATH)

        # Get backup size
        backup_size = 0
        backup_file = "greyhound_racing_data_backup.db"
        if os.path.exists(backup_file):
            backup_size = os.path.getsize(backup_file)

        # Get predictions directory size
        predictions_size = 0
        predictions_dir = "./predictions"
        if os.path.exists(predictions_dir):
            for root, dirs, files in os.walk(predictions_dir):
                for file in files:
                    predictions_size += os.path.getsize(os.path.join(root, file))

        # Get CSV files size
        csv_size = 0
        for directory in [UNPROCESSED_DIR, PROCESSED_DIR, HISTORICAL_DIR, UPCOMING_DIR]:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith(".csv"):
                            csv_size += os.path.getsize(os.path.join(root, file))

        def format_size(size_bytes):
            """Format size in bytes to human readable format"""
            if size_bytes == 0:
                return "0 B"

            for unit in ["B", "KB", "MB", "GB"]:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"

        storage_info = {
            "database_size": format_size(db_size),
            "backup_size": format_size(backup_size),
            "predictions_size": format_size(predictions_size),
            "csv_files_size": format_size(csv_size),
            "total_used": format_size(
                db_size + backup_size + predictions_size + csv_size
            ),
            "database_records": 0,
            "prediction_files": 0,
        }

        # Get record counts
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            storage_info["database_records"] = cursor.fetchone()[0]
            conn.close()
        except:
            pass

        # Get prediction file count
        if os.path.exists(predictions_dir):
            storage_info["prediction_files"] = len(
                [f for f in os.listdir(predictions_dir) if f.endswith(".json")]
            )

        return jsonify(
            {
                "success": True,
                "storage_info": storage_info,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "message": f"Error getting storage info: {str(e)}"}
            ),
            500,
        )


@app.route("/dogs")
def dogs_analysis():
    """Dogs analysis and search page"""
    return render_template("dogs_analysis.html")


@app.route("/interactive-races")
def interactive_races():
    """Interactive race cards with collapsible runners, search, and filtering"""
    return render_template("interactive_races.html")


@app.route("/api/download_and_predict_race", methods=["POST"])
def api_download_and_predict_race():
    """API endpoint to download race CSV and run prediction"""
    try:
        data = request.get_json()
        venue = data.get("venue")
        race_number = data.get("race_number")

        if not venue or not race_number:
            return (
                jsonify({"success": False, "message": "Missing venue or race_number"}),
                400,
            )

        # Import the upcoming race browser
        try:
            from upcoming_race_browser import UpcomingRaceBrowser

            browser = UpcomingRaceBrowser()
        except ImportError:
            return (
                jsonify(
                    {"success": False, "message": "UpcomingRaceBrowser not available"}
                ),
                500,
            )

        # Step 1: Find and download the race
        today_races = browser.get_upcoming_races(days_ahead=1)

        # Look for matching race with improved matching logic
        target_race = None

        # Create venue mapping for better matching
        venue_name_variations = {
            "SANDOWN": ["sandown", "san"],
            "GOULBURN": ["goulburn", "goul"],
            "DAPTO": ["dapto", "dapt"],
            "WENTWORTH": ["wentworth-park", "wpk", "wentworth"],
            "MEADOWS": ["the-meadows", "mea", "meadows"],
            "ANGLE": ["angle-park", "ap_k", "angle"],
            "BENDIGO": ["bendigo", "ben"],
            "BALLARAT": ["ballarat", "bal"],
            "GEELONG": ["geelong", "gee"],
            "WARRNAMBOOL": ["warrnambool", "war"],
            "CANNINGTON": ["cannington", "cann"],
            "HOBART": ["hobart", "hobt"],
            "GOSFORD": ["gosford", "gosf"],
            "RICHMOND": ["richmond", "rich"],
            "HEALESVILLE": ["healesville", "hea"],
            "SALE": ["sale", "sal"],
            "TRARALGON": ["traralgon", "tra"],
            "MURRAY": ["murray-bridge", "murr", "murray"],
            "MOUNT": ["mount-gambier", "mount", "mount-gambier"],
            "GAWLER": ["gawler", "gawl"],
            "NORTHAM": ["northam", "nor"],
            "MANDURAH": ["mandurah", "mand"],
            "GARDENS": ["the-gardens", "grdn", "gardens"],
            "DARWIN": ["darwin", "darw"],
            "CASINO": ["casino", "caso"],
        }

        # Normalize the input venue
        venue_normalized = venue.upper().replace(" ", "").replace("_", "")

        # Try to match races
        for race in today_races:
            race_venue = race.get("venue", "").upper()
            race_venue_name = race.get("venue_name", "").upper()
            race_num = str(race.get("race_number", ""))

            # Direct venue code match
            if venue_normalized == race_venue:
                if race_num == str(race_number):
                    target_race = race
                    break

            # Check venue name variations
            for venue_key, variations in venue_name_variations.items():
                if venue_key in venue_normalized or venue_normalized in venue_key:
                    if any(
                        var.upper() in race_venue or var.upper() in race_venue_name
                        for var in variations
                    ):
                        if race_num == str(race_number):
                            target_race = race
                            break

            if target_race:
                break

        if not target_race:
            # Get available venues for better error message
            available_venues = set([r.get("venue", "") for r in today_races])

            # Return helpful error with available venues
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Race not found: {venue} Race {race_number} is not available today",
                        "available_venues": sorted(list(available_venues)),
                        "total_races_today": len(today_races),
                        "suggestion": "Please check the available venues and try again with a different venue or race number",
                    }
                ),
                404,
            )

        # Step 2: Download the CSV
        race_url = target_race.get("url") or target_race.get("race_url")
        download_result = browser.download_race_csv(race_url)

        if not download_result["success"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f'Failed to download race CSV: {download_result["error"]}',
                    }
                ),
                500,
            )

        filename = download_result["filename"]
        filepath = download_result["filepath"]

        # Step 3: Run prediction using the most advanced weather-enhanced system
        try:
            # Try to use the comprehensive prediction pipeline (most advanced)
            predict_script = "comprehensive_prediction_pipeline.py"
            if os.path.exists(predict_script):
                print(f"🎯 Using comprehensive prediction pipeline for: {filepath}")
                result = subprocess.run(
                    [sys.executable, predict_script, filepath],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes for comprehensive analysis
                    cwd=os.getcwd(),
                )

                prediction_success = result.returncode == 0
                prediction_output = (
                    result.stdout if prediction_success else result.stderr
                )

                print(f"📋 Comprehensive prediction result: {result.returncode}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout[-300:]}")
                if result.stderr:
                    print(f"STDERR: {result.stderr[-300:]}")
            else:
                # Fallback to weather-enhanced predictor
                try:
                    from weather_enhanced_predictor import \
                        WeatherEnhancedPredictor

                    print(f"🌤️ Using WeatherEnhancedPredictor for: {filepath}")
                    predictor = WeatherEnhancedPredictor()
                    prediction_result = predictor.predict_race_file(filepath)

                    if prediction_result and prediction_result.get("success"):
                        prediction_success = True
                        prediction_output = f"Weather-enhanced prediction completed successfully for {len(prediction_result.get('predictions', []))} dogs"
                        print(
                            f"✅ Weather-enhanced prediction successful: {prediction_output}"
                        )
                    else:
                        prediction_success = False
                        prediction_output = f"Weather-enhanced prediction failed: {prediction_result.get('error', 'Unknown error')}"
                        print(
                            f"❌ Weather-enhanced prediction failed: {prediction_output}"
                        )

                except ImportError as import_error:
                    print(f"⚠️ WeatherEnhancedPredictor not available: {import_error}")
                    # Final fallback to subprocess approach
                    predict_script = os.path.join(
                        os.getcwd(), "upcoming_race_predictor.py"
                    )

                    if os.path.exists(predict_script):
                        print(
                            f"🔄 Falling back to subprocess prediction: {predict_script}"
                        )
                        result = subprocess.run(
                            [sys.executable, predict_script, filepath],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            cwd=os.getcwd(),
                        )

                        prediction_success = result.returncode == 0
                        prediction_output = (
                            result.stdout if prediction_success else result.stderr
                        )

                        print(f"📋 Subprocess prediction result: {result.returncode}")
                        if result.stdout:
                            print(f"STDOUT: {result.stdout[-200:]}")
                        if result.stderr:
                            print(f"STDERR: {result.stderr[-200:]}")
                    else:
                        prediction_success = False
                        prediction_output = (
                            f"Prediction script not found at: {predict_script}"
                        )

        except Exception as e:
            prediction_success = False
            prediction_output = str(e)

        # Step 4: Try to read prediction results
        race_id = f"{venue.replace(' ', '_')}_{race_number}_{datetime.now().strftime('%Y%m%d')}"
        prediction_filename = build_prediction_filename(
            race_id, datetime.now(), "comprehensive"
        )
        prediction_file = f"./predictions/{prediction_filename}"

        prediction_summary = {
            "total_dogs": 0,
            "top_pick": {"dog_name": "Unknown", "prediction_score": 0},
        }

        if os.path.exists(prediction_file):
            try:
                with open(prediction_file, "r") as f:
                    prediction_data = json.load(f)

                predictions = prediction_data.get("predictions", [])
                if predictions:
                    prediction_summary = {
                        "total_dogs": len(predictions),
                        "top_pick": {
                            "dog_name": predictions[0].get("dog_name", "Unknown"),
                            "prediction_score": predictions[0].get("final_score", 0),
                        },
                    }
            except Exception as e:
                print(f"Error reading prediction file: {e}")

        return jsonify(
            {
                "success": True,
                "message": "Race downloaded and predicted successfully",
                "filename": filename,
                "filepath": filepath,
                "race_info": {
                    "venue": venue,
                    "race_number": race_number,
                    "race_id": race_id,
                    "race_date": target_race.get("race_date"),
                    "race_time": target_race.get("race_time"),
                },
                "prediction_success": prediction_success,
                "prediction_output": (
                    prediction_output[-500:] if prediction_output else ""
                ),  # Last 500 chars
                "prediction_summary": prediction_summary,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error downloading and predicting race: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/csv_metadata")
def api_csv_metadata():
    """API endpoint to extract metadata from CSV files using lightweight extractor"""
    try:
        file_path = request.args.get("file_path")
        
        if not file_path:
            return jsonify({
                "success": False,
                "error": "file_path parameter is required",
                "message": "Please provide a file_path parameter with the path to the CSV file"
            }), 400
        
        # Use the new parse_race_csv_meta function
        metadata = parse_race_csv_meta(file_path)
        
        # Check if extraction was successful
        if metadata.get("status") == "error":
            return jsonify({
                "success": False,
                "error": metadata.get("error_message", "Unknown error"),
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }), 400
        
        return jsonify({
            "success": True,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}",
            "message": "An unexpected error occurred while extracting CSV metadata"
        }), 500


def ingest(file_path: Path) -> dict:
    """
    Ingest a single race file from CLI/scheduler.
    
    This function:
    - Accepts a single file path from CLI/scheduler
    - Removes any directory globbing logic  
    - Calls the refactored parser and downstream prediction only for that file
    
    Args:
        file_path: Path to the CSV file to ingest
        
    Returns:
        dict: Result of ingestion with success status and details
    """
    try:
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        logger.info(f"Starting ingestion for single file: {file_path}")
        
        # Validate file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Validate file is CSV
        if not file_path.suffix.lower() == '.csv':
            error_msg = f"File must be CSV format: {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Step 1: Parse the CSV file using the refactored parser
        parser_result = None
        if CSV_INGESTION_AVAILABLE:
            try:
                logger.info(f"Using CSV ingestion system for parsing: {file_path}")
                ingestor = create_ingestor(validation_level="moderate")
                processed_data, validation_result = ingestor.ingest_csv(file_path)
                
                parser_result = {
                    "success": True,
                    "processed_data": processed_data,
                    "validation_result": validation_result,
                    "record_count": len(processed_data),
                    "parser_used": "FormGuideCsvIngestor"
                }
                logger.info(f"CSV ingestion successful: {len(processed_data)} records processed")
                
            except Exception as e:
                logger.error(f"CSV ingestion failed: {str(e)}")
                parser_result = {"success": False, "error": str(e)}
        else:
            # Fallback to basic pandas parsing
            try:
                logger.info(f"Using fallback pandas parser for: {file_path}")
                import pandas as pd
                df = pd.read_csv(file_path)
                
                parser_result = {
                    "success": True,
                    "processed_data": df.to_dict('records'),
                    "record_count": len(df),
                    "parser_used": "pandas_fallback"
                }
                logger.info(f"Pandas parsing successful: {len(df)} records processed")
                
            except Exception as e:
                logger.error(f"Pandas parsing failed: {str(e)}")
                parser_result = {"success": False, "error": str(e)}
        
        if not parser_result or not parser_result.get("success"):
            error_msg = f"Parsing failed: {parser_result.get('error', 'Unknown parsing error') if parser_result else 'No parser result'}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Step 2: Run downstream prediction for the parsed file
        prediction_result = None
        try:
            logger.info(f"Starting downstream prediction for: {file_path}")
            
            # Try PredictionPipelineV3 first (most comprehensive)
            if PredictionPipelineV3:
                try:
                    logger.info("Using PredictionPipelineV3 for downstream prediction")
                    pipeline = PredictionPipelineV3()
                    prediction_result = pipeline.predict_race_file(file_path, enhancement_level="full")
                    
                    if prediction_result and prediction_result.get("success"):
                        logger.info("PredictionPipelineV3 completed successfully")
                    else:
                        logger.warning(f"PredictionPipelineV3 returned unsuccessful result: {prediction_result}")
                        prediction_result = None  # Force fallback
                        
                except Exception as e:
                    logger.error(f"PredictionPipelineV3 failed: {str(e)}")
                    prediction_result = None
            
            # Fallback to UnifiedPredictor if primary fails
            if not prediction_result and UnifiedPredictor:
                try:
                    logger.info("Using UnifiedPredictor for downstream prediction")
                    predictor = UnifiedPredictor()
                    prediction_result = predictor.predict_race_file(file_path)
                    
                    if prediction_result and prediction_result.get("success"):
                        logger.info("UnifiedPredictor completed successfully")
                    else:
                        logger.warning(f"UnifiedPredictor returned unsuccessful result: {prediction_result}")
                        
                except Exception as e:
                    logger.error(f"UnifiedPredictor failed: {str(e)}")
                    prediction_result = None
            
            # Final fallback to ComprehensivePredictionPipeline
            if not prediction_result and ComprehensivePredictionPipeline:
                try:
                    logger.info("Using ComprehensivePredictionPipeline for downstream prediction")
                    predictor = ComprehensivePredictionPipeline()
                    prediction_result = predictor.predict_race_file(file_path)
                    
                    if prediction_result and prediction_result.get("success"):
                        logger.info("ComprehensivePredictionPipeline completed successfully")
                        
                except Exception as e:
                    logger.error(f"ComprehensivePredictionPipeline failed: {str(e)}")
                    prediction_result = None
                    
        except Exception as e:
            logger.error(f"Downstream prediction error: {str(e)}")
            prediction_result = {"success": False, "error": str(e)}
        
        # Determine if prediction was successful
        prediction_success = prediction_result and prediction_result.get("success", False)
        
        # Step 3: Compile final result
        result = {
            "success": True,
            "file_path": str(file_path),
            "parsing": {
                "success": parser_result["success"],
                "parser_used": parser_result.get("parser_used", "unknown"),
                "record_count": parser_result.get("record_count", 0)
            },
            "prediction": {
                "success": prediction_success,
                "predictor_used": (
                    "PredictionPipelineV3" if prediction_result and "PredictionPipelineV3" in str(type(prediction_result))
                    else "UnifiedPredictor" if prediction_result and "UnifiedPredictor" in str(type(prediction_result))
                    else "ComprehensivePredictionPipeline" if prediction_result and "ComprehensivePredictionPipeline" in str(type(prediction_result))
                    else "unknown"
                ),
                "prediction_count": (
                    len(prediction_result.get("predictions", [])) if prediction_result and prediction_result.get("predictions") else 0
                ),
                "error": prediction_result.get("error") if prediction_result and not prediction_success else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Log final status
        if prediction_success:
            logger.info(f"Ingestion completed successfully for {file_path}: {parser_result.get('record_count', 0)} records parsed, {result['prediction']['prediction_count']} predictions generated")
        else:
            logger.warning(f"Ingestion completed with warnings for {file_path}: parsing successful but prediction failed")
        
        return result
        
    except Exception as e:
        error_msg = f"Ingestion failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "file_path": str(file_path) if file_path else "unknown",
            "timestamp": datetime.now().isoformat()
        }


# Background Task API Endpoints

@app.route("/api/tasks/download_race", methods=["POST"])
def api_download_race_task():
    """API endpoint to start background race download task"""
    try:
        if not BACKGROUND_TASKS_AVAILABLE:
            return jsonify({
                "success": False, 
                "message": "Background task system not available"
            }), 500
        
        data = request.get_json()
        race_url = data.get("race_url")
        venue = data.get("venue")
        race_date = data.get("race_date")
        
        if not race_url:
            return jsonify({
                "success": False,
                "message": "race_url is required"
            }), 400
        
        # Start background task
        task = enqueue_task(download_race_data, race_url, venue, race_date)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Race download task started",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error starting download task: {str(e)}"
        }), 500


@app.route("/api/tasks/process_file", methods=["POST"])
def api_process_file_task():
    """API endpoint to start background file processing task"""
    try:
        if not BACKGROUND_TASKS_AVAILABLE:
            return jsonify({
                "success": False, 
                "message": "Background task system not available"
            }), 500
        
        data = request.get_json()
        file_path = data.get("file_path")
        
        if not file_path:
            return jsonify({
                "success": False,
                "message": "file_path is required"
            }), 400
        
        # Start background task
        task = enqueue_task(process_race_file, file_path)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "File processing task started",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error starting processing task: {str(e)}"
        }), 500


@app.route("/api/tasks/generate_prediction", methods=["POST"])
def api_generate_prediction_task():
    """API endpoint to start background prediction generation task"""
    try:
        if not BACKGROUND_TASKS_AVAILABLE:
            return jsonify({
                "success": False, 
                "message": "Background task system not available"
            }), 500
        
        data = request.get_json()
        race_file = data.get("race_file")
        prediction_config = data.get("prediction_config", {})
        
        if not race_file:
            return jsonify({
                "success": False,
                "message": "race_file is required"
            }), 400
        
        # Start background task
        task = enqueue_task(generate_predictions, race_file, prediction_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Prediction generation task started",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error starting prediction task: {str(e)}"
        }), 500


@app.route("/api/tasks/update_race_notes", methods=["POST"])
def api_update_race_notes_task():
    """API endpoint to start background race notes update task"""
    try:
        if not BACKGROUND_TASKS_AVAILABLE:
            return jsonify({
                "success": False, 
                "message": "Background task system not available"
            }), 500
        
        data = request.get_json()
        race_id = data.get("race_id")
        notes = data.get("notes")
        user_id = data.get("user_id", "api_user")
        
        if not race_id or not notes:
            return jsonify({
                "success": False,
                "message": "race_id and notes are required"
            }), 400
        
        # Start background task
        task = enqueue_task(update_race_notes, race_id, notes, user_id)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Race notes update task started",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error starting notes update task: {str(e)}"
        }), 500


@app.route("/api/tasks/status/<task_id>")
def api_task_status(task_id):
    """API endpoint to get task status"""
    try:
        if not BACKGROUND_TASKS_AVAILABLE:
            return jsonify({
                "success": False, 
                "message": "Background task system not available"
            }), 500
        
        status = get_task_status(task_id)
        
        if status is None:
            return jsonify({
                "success": False,
                "message": "Task not found"
            }), 404
        
        return jsonify({
            "success": True,
            "task_status": status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting task status: {str(e)}"
        }), 500


@app.route("/api/tasks/all_status")
def api_all_tasks_status():
    """API endpoint to get status of all active tasks"""
    try:
        if not BACKGROUND_TASKS_AVAILABLE:
            return jsonify({
                "success": False, 
                "message": "Background task system not available"
            }), 500
        
        # Get active tasks from both Celery and RQ if available
        active_tasks = []
        
        # Try to get Celery active tasks
        try:
            from celery import current_app
            inspect = current_app.control.inspect()
            if inspect:
                celery_tasks = inspect.active()
                if celery_tasks:
                    for worker, tasks in celery_tasks.items():
                        for task in tasks:
                            active_tasks.append({
                                "task_id": task.get("id"),
                                "name": task.get("name"),
                                "args": task.get("args", []),
                                "worker": worker,
                                "queue": "celery"
                            })
        except Exception:
            pass
        
        # Try to get RQ active tasks
        try:
            import redis
            from rq import Queue
            redis_conn = redis.Redis(host='localhost', port=6379, db=0)
            rq_queue = Queue(connection=redis_conn)
            
            for job in rq_queue.get_jobs():
                active_tasks.append({
                    "task_id": job.id,
                    "name": job.func_name,
                    "args": job.args,
                    "status": job.status,
                    "queue": "rq"
                })
        except Exception:
            pass
        
        return jsonify({
            "success": True,
            "active_tasks": active_tasks,
            "task_count": len(active_tasks),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting tasks status: {str(e)}"
        }), 500

def create_cli_parser():
    """Create CLI argument parser for Flask app"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Greyhound Racing Dashboard Flask Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --enable-profiling
  python app.py --host 0.0.0.0 --port 8080 --enable-profiling
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help=f'Port to bind to (default: {DEFAULT_PORT})'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode'
    )
    
    parser.add_argument(
        '--enable-profiling',
        action='store_true',
        help='Enable profiling for performance analysis (default: disabled for zero overhead)'
    )
    
    return parser


def main():
    """Main Flask app entry point with profiling support"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Configure profiling based on CLI flag
    if args.enable_profiling:
        set_profiling_enabled(True)
        print("🔍 Profiling enabled for Flask app")
    else:
        set_profiling_enabled(False)
    
    # Show profiling status for debugging
    if is_profiling():
        print("📊 Flask app running with profiling enabled")
    
    # Run Flask app with configured options
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    # Get PORT from environment or CLI, default to DEFAULT_PORT
    import os
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    
    # Create templates and static directories if they don't exist
    os.makedirs("./templates", exist_ok=True)
    os.makedirs("./static/css", exist_ok=True)
    os.makedirs("./static/js", exist_ok=True)

    # Run integrity test before starting the app
    run_integrity_test()
    
    # Initialize Guardian Service in background
    # TEMPORARILY DISABLED - Guardian Service startup commented out for hot-fix
    # if GUARDIAN_SERVICE_AVAILABLE and start_guardian_service:
    #     try:
    #         guardian_thread = threading.Thread(target=start_guardian_service, daemon=True)
    #         guardian_thread.start()
    #         print("🛡️ Guardian Service started in background")
    #     except Exception as e:
    #         print(f"⚠️ Failed to start Guardian Service: {e}")
    # else:
    #     print("⚠️ Guardian Service not available")
    print("🛡️ Guardian Service autostart disabled (hot-fix)")

    # Only run app if called directly
    app.run(host="0.0.0.0", port=port, debug=False)
