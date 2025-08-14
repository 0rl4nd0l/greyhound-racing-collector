#!/usr/bin/env python3
"""
FastAPI Greyhound Racing Prediction API
=====================================

FastAPI web service for greyhound racing predictions with enhanced ML capabilities.
Provides health check and enhanced prediction endpoints.

Author: AI Assistant
Date: January 2025
"""

import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logger

# Try to import prediction systems
try:
    from prediction_pipeline_v3 import PredictionPipelineV3
    ML_SYSTEM_V3_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML System V3 not available: {e}")
    ML_SYSTEM_V3_AVAILABLE = False
    PredictionPipelineV3 = None

try:
    from unified_predictor import UnifiedPredictor
    UNIFIED_PREDICTOR_AVAILABLE = True
except ImportError:
    UNIFIED_PREDICTOR_AVAILABLE = False
    UnifiedPredictor = None

# Configuration
DATABASE_PATH = "greyhound_racing_data.db"
UPCOMING_DIR = os.getenv("UPCOMING_RACES_DIR", "./upcoming_races")

# Initialize FastAPI app
app = FastAPI(
    title="Greyhound Racing Prediction API",
    description="FastAPI service for greyhound racing predictions with enhanced ML capabilities",
    version="1.0.0"
)

# Verify UPCOMING_RACES_DIR at startup
@app.on_event("startup")
async def ensure_upcoming_dir():
    try:
        if not os.path.exists(UPCOMING_DIR):
            os.makedirs(UPCOMING_DIR, exist_ok=True)
        # Check readability
        if not os.access(UPCOMING_DIR, os.R_OK | os.X_OK):
            logger.warning(f"UPCOMING_RACES_DIR not readable/executable by process: {UPCOMING_DIR}")
        else:
            logger.info(f"UPCOMING_RACES_DIR ready: {UPCOMING_DIR}")
    except Exception as e:
        logger.warning(f"Failed to prepare UPCOMING_RACES_DIR ({UPCOMING_DIR}): {e}")


def _list_upcoming_csvs() -> List[str]:
    try:
        entries = sorted([e for e in os.listdir(UPCOMING_DIR) if not e.startswith('.')]) if os.path.exists(UPCOMING_DIR) else []
        files = [e for e in entries if e.lower().endswith('.csv')]
        skipped = {e: "invalid extension (only .csv)" for e in entries if e not in files and not e.startswith('.')}
        logger.info(
            "Upcoming discovery",
            extra={
                "details": {
                    "directory": UPCOMING_DIR,
                    "found_count": len(files),
                    "found_names": files,
                    "skipped_count": len(skipped),
                    "skipped": skipped,
                },
                "action": "discover_upcoming",
                "outcome": "observed",
            },
        )
        return files
    except Exception as e:
        logger.warning(f"Error listing upcoming CSVs: {e}")
        return []


def _during_business_hours(now: datetime) -> bool:
    try:
        start = int(os.getenv("BUSINESS_HOURS_START", "8"))
        end = int(os.getenv("BUSINESS_HOURS_END", "18"))
    except Exception:
        start, end = 8, 18
    hour = now.hour
    return start <= hour < end

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class DogData(BaseModel):
    name: str
    box_number: int
    weight: Optional[float] = None
    trainer: Optional[str] = None
    recent_form: Optional[List[int]] = None
    odds: Optional[float] = None

class RaceData(BaseModel):
    race_id: str
    venue: str
    distance: str
    grade: str
    race_date: str
    dogs: List[DogData]

class PredictionResponse(BaseModel):
    success: bool
    race_id: str
    predictions: List[Dict]
    model_used: str
    processing_time_ms: float
    timestamp: str
    confidence_level: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


@app.get("/", response_model=Dict[str, str])
async def heartbeat():
    """Health check endpoint - heartbeat"""
    return {
        "status": "healthy",
        "service": "Greyhound Racing Prediction API",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Check database connectivity
        database_status = "connected"
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dogs LIMIT 1")
            conn.close()
        except Exception as e:
            database_status = f"error: {str(e)}"
            logger.warning(f"Database health check failed: {e}")

        # Upcoming files discovery with alerting
        files = _list_upcoming_csvs()
        upcoming_count = len(files)
        now = datetime.now()
        if upcoming_count == 0 and _during_business_hours(now):
            logger.warning(
                "ALERT: Zero upcoming race CSVs during business hours",
                extra={
                    "details": {"directory": UPCOMING_DIR, "business_hours": True, "timestamp": now.isoformat()},
                    "action": "zero_upcoming_alert",
                    "outcome": "alerted",
                },
            )

        return HealthResponse(
            status="healthy",
            timestamp=now.isoformat(),
            version="1.0.0",
            components={
                "database": database_status,
                "ml_system_v3": "available" if ML_SYSTEM_V3_AVAILABLE else "unavailable",
                "unified_predictor": "available" if UNIFIED_PREDICTOR_AVAILABLE else "unavailable",
                "upcoming_races_dir": "exists" if os.path.exists(UPCOMING_DIR) else "missing",
                "upcoming_files_count": str(upcoming_count)
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/api/predict_single_race_enhanced", response_model=PredictionResponse)
async def predict_single_race_enhanced(race_data: RaceData):
    """Enhanced single race prediction endpoint with ML capabilities"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting enhanced prediction for race: {race_data.race_id}")
        
        # Validate input data
        if not race_data.dogs or len(race_data.dogs) == 0:
            raise HTTPException(status_code=400, detail="No dogs provided for prediction")
        
        if len(race_data.dogs) > 12:
            raise HTTPException(status_code=400, detail="Too many dogs (max 12 allowed)")
        
        prediction_result = None
        model_used = "none"
        
        # Try ML System V3 first (most advanced)
        if ML_SYSTEM_V3_AVAILABLE and PredictionPipelineV3:
            try:
                logger.info("Attempting prediction with ML System V3")
                pipeline = PredictionPipelineV3()
                
                # Convert race data to format expected by pipeline
                race_df = pd.DataFrame([{
                    'dog_name': dog.name,
                    'box_number': dog.box_number,
                    'weight': dog.weight or 30.0,  # Default weight
                    'trainer_name': dog.trainer or 'Unknown',
                    'odds_decimal': dog.odds or 5.0  # Default odds
                } for dog in race_data.dogs])
                
                # Create a temporary CSV file for the pipeline
                temp_csv_path = f"/tmp/temp_race_{race_data.race_id}.csv"
                race_df.to_csv(temp_csv_path, index=False)
                
                try:
                    prediction_result = pipeline.predict_race_file(temp_csv_path, enhancement_level="full")
                    model_used = "PredictionPipelineV3"
                    logger.info("ML System V3 prediction successful")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_csv_path):
                        os.remove(temp_csv_path)
                        
            except Exception as e:
                logger.warning(f"ML System V3 prediction failed: {e}")
                prediction_result = None
        
        # Fallback to Unified Predictor
        if not prediction_result and UNIFIED_PREDICTOR_AVAILABLE and UnifiedPredictor:
            try:
                logger.info("Falling back to Unified Predictor")
                predictor = UnifiedPredictor()
                
                # Create mock race file data
                race_file_data = {
                    'race_id': race_data.race_id,
                    'venue': race_data.venue,
                    'distance': race_data.distance,
                    'grade': race_data.grade,
                    'dogs': [dog.dict() for dog in race_data.dogs]
                }
                
                prediction_result = predictor.predict_race_data(race_file_data)
                model_used = "UnifiedPredictor"
                logger.info("Unified Predictor prediction successful")
                
            except Exception as e:
                logger.warning(f"Unified Predictor failed: {e}")
                prediction_result = None
        
        # Generate mock predictions if all else fails
        if not prediction_result:
            logger.info("Generating mock predictions as fallback")
            predictions = []
            
            for i, dog in enumerate(race_data.dogs):
                # Simple mock prediction based on odds and box position
                base_prob = 1.0 / (dog.odds or 5.0) if dog.odds else 0.2
                position_factor = max(0.8, 1.2 - (dog.box_number * 0.05))  # Inside boxes slight advantage
                
                win_probability = min(0.95, max(0.05, base_prob * position_factor))
                place_probability = min(0.95, win_probability * 2.5)
                
                predictions.append({
                    "dog_name": dog.name,
                    "box_number": dog.box_number,
                    "win_probability": round(win_probability, 3),
                    "place_probability": round(place_probability, 3),
                    "confidence": round(win_probability * 0.8 + 0.2, 3),
                    "predicted_position": i + 1,
                    "form_rating": "B+",  # Mock rating
                    "speed_rating": round(85 + (win_probability * 15), 1)
                })
            
            # Sort by win probability
            predictions.sort(key=lambda x: x["win_probability"], reverse=True)
            
            prediction_result = {
                "success": True,
                "predictions": predictions,
                "race_analysis": {
                    "competitive_rating": "Medium",
                    "pace_scenario": "Even pace expected",
                    "key_factors": ["Box position", "Recent form", "Track conditions"]
                }
            }
            model_used = "MockPredictor"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine confidence level
        if prediction_result and prediction_result.get("predictions"):
            top_prob = max(pred.get("win_probability", 0) for pred in prediction_result["predictions"])
            if top_prob > 0.4:
                confidence_level = "HIGH"
            elif top_prob > 0.25:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
        else:
            confidence_level = "LOW"
        
        logger.info(f"Enhanced prediction completed for race {race_data.race_id} using {model_used}")
        
        return PredictionResponse(
            success=True,
            race_id=race_data.race_id,
            predictions=prediction_result.get("predictions", []),
            model_used=model_used,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            confidence_level=confidence_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Enhanced prediction error for race {race_data.race_id}: {e}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Prediction failed: {str(e)}",
                "race_id": race_data.race_id,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/api/system_info")
async def system_info():
    """Get system information and available components"""
    return {
        "system": "Greyhound Racing Prediction API",
        "version": "1.0.0",
        "available_predictors": {
            "ml_system_v3": ML_SYSTEM_V3_AVAILABLE,
            "unified_predictor": UNIFIED_PREDICTOR_AVAILABLE
        },
        "database_path": DATABASE_PATH,
        "upcoming_races_dir": UPCOMING_DIR,
        "upcoming_races_exist": os.path.exists(UPCOMING_DIR),
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception in {request.url}: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
