#!/usr/bin/env python3
"""
Prediction Adapters - Standardized Thin Wrappers
===============================================

This module provides thin wrapper adapters for different prediction models:
- V3Adapter: Uses PredictionPipelineV3 with MLSystemV4 backend
- V3SAdapter: Maps to special-case/simple predictor if available, otherwise falls back to V3Adapter
- V4Adapter: Uses MLSystemV4.predict_race on preprocessed DataFrame

All adapters return a standardized structure:
{
    race_id: str,
    predictions: [
        {
            dog: str,
            win_prob_norm: float,
            raw_prob: float
        }
    ],
    metadata: dict
}
"""

import logging
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import prediction systems
from prediction_pipeline_v3 import PredictionPipelineV3
from ml_system_v4 import MLSystemV4

# Try to import simple/special-case predictor for V3S
try:
    from unified_predictor import UnifiedPredictor
    SIMPLE_PREDICTOR_AVAILABLE = True
except ImportError:
    SIMPLE_PREDICTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class StandardizedResult:
    """Helper class to ensure consistent result format across all adapters."""
    
    @staticmethod
    def create_result(race_id: str, predictions: List[Dict[str, Any]], 
                     metadata: Dict[str, Any], success: bool = True, 
                     error: Optional[str] = None) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        result = {
            "race_id": race_id,
            "predictions": predictions,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                **metadata
            }
        }
        
        if error:
            result["metadata"]["error"] = error
            
        return result
    
    @staticmethod
    def normalize_probabilities(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply group softmax normalization to win probabilities."""
        import numpy as np
        
        if not predictions:
            return predictions
            
        # Extract raw probabilities
        raw_probs = [pred.get("raw_prob", 0.0) for pred in predictions]
        
        # Apply softmax normalization
        if sum(raw_probs) > 0:
            # Use temperature scaling for better calibration
            temperature = 1.2
            exp_probs = np.exp(np.array(raw_probs) / temperature)
            normalized_probs = exp_probs / np.sum(exp_probs)
        else:
            # Equal probabilities if all raw probs are zero
            normalized_probs = np.ones(len(predictions)) / len(predictions)
        
        # Update predictions with normalized probabilities
        for i, pred in enumerate(predictions):
            pred["win_prob_norm"] = float(normalized_probs[i])
            
        return predictions


class V3Adapter:
    """
    V3 Adapter - calls PredictionPipelineV3(...).ml_system.predict per dog.
    
    This adapter uses the comprehensive V3 prediction pipeline which internally
    uses MLSystemV4 but provides enhanced features and fallback mechanisms.
    """
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.pipeline = None
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the PredictionPipelineV3."""
        try:
            self.pipeline = PredictionPipelineV3(self.db_path)
            logger.info("✅ V3Adapter: PredictionPipelineV3 initialized successfully")
        except Exception as e:
            logger.error(f"❌ V3Adapter: Failed to initialize PredictionPipelineV3: {e}")
            self.pipeline = None
    
    def predict_race(self, race_file_path: str) -> Dict[str, Any]:
        """
        Predict race using PredictionPipelineV3 and convert to standardized format.
        
        Args:
            race_file_path: Path to the race CSV file
            
        Returns:
            Standardized prediction result dictionary
        """
        if not self.pipeline:
            return StandardizedResult.create_result(
                race_id=os.path.basename(race_file_path),
                predictions=[],
                metadata={"adapter": "V3Adapter", "method": "error"},
                success=False,
                error="Pipeline not initialized"
            )
        
        try:
            # Use the comprehensive prediction pipeline
            pipeline_result = self.pipeline.predict_race_file(race_file_path)
            
            if not pipeline_result.get("success", False):
                return StandardizedResult.create_result(
                    race_id=os.path.basename(race_file_path),
                    predictions=[],
                    metadata={
                        "adapter": "V3Adapter",
                        "method": "pipeline_error",
                        "original_error": pipeline_result.get("error", "Unknown error")
                    },
                    success=False,
                    error=pipeline_result.get("error", "Pipeline prediction failed")
                )
                
            # Convert pipeline predictions to standardized format
            standardized_predictions = []
            
            for pred in pipeline_result.get("predictions", []):
                standardized_pred = {
                    "dog": pred.get("dog_name", "Unknown"),
                    "raw_prob": pred.get("win_probability", 0.0),
                    "win_prob_norm": 0.0,  # Will be set by normalization
                    "box_number": pred.get("box_number", 0),
                    "method": pred.get("prediction_method", "ml_system_v3"),
                    "confidence": pred.get("confidence", 0.0) if "confidence" in pred else None
                }
                standardized_predictions.append(standardized_pred)
            
            # Apply group normalization
            standardized_predictions = StandardizedResult.normalize_probabilities(standardized_predictions)
            
            # Sort by normalized probability (highest first)
            standardized_predictions.sort(key=lambda x: x["win_prob_norm"], reverse=True)
            
            metadata = {
                "adapter": "V3Adapter",
                "method": pipeline_result.get("prediction_method", "unknown"),
                "prediction_tier": pipeline_result.get("prediction_tier", "unknown"),
                "total_dogs": len(standardized_predictions),
                "fallback_reasons": pipeline_result.get("fallback_reasons", [])
            }
            
            return StandardizedResult.create_result(
                race_id=os.path.basename(race_file_path),
                predictions=standardized_predictions,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ V3Adapter: Error predicting race {race_file_path}: {e}")
            return StandardizedResult.create_result(
                race_id=os.path.basename(race_file_path),
                predictions=[],
                metadata={"adapter": "V3Adapter", "method": "exception"},
                success=False,
                error=str(e)
            )


class V3SAdapter:
    """
    V3S Adapter - maps to special-case/simple predictor if available, otherwise same as V3Adapter.
    
    This adapter first tries to use a simple/special-case predictor for faster predictions.
    If not available, it falls back to the full V3Adapter functionality.
    """
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.simple_predictor = None
        self.fallback_adapter = None
        self._initialize_predictors()
        
    def _initialize_predictors(self):
        """Initialize simple predictor and fallback adapter."""
        # Try to initialize simple predictor first
        if SIMPLE_PREDICTOR_AVAILABLE:
            try:
                self.simple_predictor = UnifiedPredictor()
                logger.info("✅ V3SAdapter: Simple predictor (UnifiedPredictor) initialized")
            except Exception as e:
                logger.warning(f"⚠️ V3SAdapter: Simple predictor initialization failed: {e}")
                self.simple_predictor = None
        
        # Initialize fallback adapter
        try:
            self.fallback_adapter = V3Adapter(self.db_path)
            logger.info("✅ V3SAdapter: Fallback V3Adapter initialized")
        except Exception as e:
            logger.error(f"❌ V3SAdapter: Fallback adapter initialization failed: {e}")
            self.fallback_adapter = None
    
    def predict_race(self, race_file_path: str) -> Dict[str, Any]:
        """
        Predict race using simple predictor if available, otherwise fallback to V3Adapter.
        
        Args:
            race_file_path: Path to the race CSV file
            
        Returns:
            Standardized prediction result dictionary
        """
        race_id = os.path.basename(race_file_path)
        
        # Try simple predictor first
        if self.simple_predictor:
            try:
                logger.info(f"🚀 V3SAdapter: Trying simple predictor for {race_id}")
                simple_result = self.simple_predictor.predict_race_file(race_file_path)
                
                if simple_result.get("success", False):
                    # Convert simple predictor result to standardized format
                    standardized_predictions = []
                    
                    for pred in simple_result.get("predictions", []):
                        standardized_pred = {
                            "dog": pred.get("dog_name", "Unknown"),
                            "raw_prob": pred.get("win_probability", 0.0),
                            "win_prob_norm": 0.0,  # Will be set by normalization
                            "box_number": pred.get("box_number", 0),
                            "method": "simple_predictor",
                            "confidence": pred.get("confidence", 0.0) if "confidence" in pred else None
                        }
                        standardized_predictions.append(standardized_pred)
                    
                    # Apply group normalization
                    standardized_predictions = StandardizedResult.normalize_probabilities(standardized_predictions)
                    
                    # Sort by normalized probability
                    standardized_predictions.sort(key=lambda x: x["win_prob_norm"], reverse=True)
                    
                    metadata = {
                        "adapter": "V3SAdapter",
                        "method": "simple_predictor",
                        "predictor_type": "UnifiedPredictor",
                        "total_dogs": len(standardized_predictions)
                    }
                    
                    logger.info(f"✅ V3SAdapter: Simple predictor succeeded for {race_id}")
                    return StandardizedResult.create_result(
                        race_id=race_id,
                        predictions=standardized_predictions,
                        metadata=metadata
                    )
                    
                else:
                    logger.warning(f"⚠️ V3SAdapter: Simple predictor failed for {race_id}, falling back to V3Adapter")
                    
            except Exception as e:
                logger.warning(f"⚠️ V3SAdapter: Simple predictor exception for {race_id}: {e}, falling back to V3Adapter")
        
        # Fallback to V3Adapter
        if self.fallback_adapter:
            logger.info(f"🔄 V3SAdapter: Using fallback V3Adapter for {race_id}")
            fallback_result = self.fallback_adapter.predict_race(race_file_path)
            
            # Update metadata to indicate fallback was used
            if "metadata" in fallback_result:
                fallback_result["metadata"]["adapter"] = "V3SAdapter"
                fallback_result["metadata"]["fallback_used"] = True
                fallback_result["metadata"]["fallback_reason"] = "simple_predictor_unavailable" if not self.simple_predictor else "simple_predictor_failed"
            
            return fallback_result
        
        # If both simple predictor and fallback failed
        return StandardizedResult.create_result(
            race_id=race_id,
            predictions=[],
            metadata={"adapter": "V3SAdapter", "method": "error"},
            success=False,
            error="Both simple predictor and fallback adapter unavailable"
        )


class V4Adapter:
    """
    V4 Adapter - calls MLSystemV4.predict_race on a pre-processed DataFrame.
    
    This adapter directly uses MLSystemV4's predict_race method, which provides
    temporal leakage protection and advanced calibration features.
    """
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ml_system = None
        self._initialize_ml_system()
        
    def _initialize_ml_system(self):
        """Initialize MLSystemV4."""
        try:
            self.ml_system = MLSystemV4(self.db_path)
            logger.info("✅ V4Adapter: MLSystemV4 initialized successfully")
        except Exception as e:
            logger.error(f"❌ V4Adapter: Failed to initialize MLSystemV4: {e}")
            self.ml_system = None
    
    def predict_race(self, race_file_path: str) -> Dict[str, Any]:
        """
        Predict race using MLSystemV4.predict_race on preprocessed DataFrame.
        
        Args:
            race_file_path: Path to the race CSV file
            
        Returns:
            Standardized prediction result dictionary
        """
        if not self.ml_system:
            return StandardizedResult.create_result(
                race_id=os.path.basename(race_file_path),
                predictions=[],
                metadata={"adapter": "V4Adapter", "method": "error"},
                success=False,
                error="MLSystemV4 not initialized"
            )
        
        try:
            # Load and preprocess the CSV file
            race_data = self._preprocess_csv(race_file_path)
            if race_data.empty:
                return StandardizedResult.create_result(
                    race_id=os.path.basename(race_file_path),
                    predictions=[],
                    metadata={"adapter": "V4Adapter", "method": "preprocessing_error"},
                    success=False,
                    error="Failed to preprocess CSV data"
                )
            
            race_id = os.path.basename(race_file_path).replace('.csv', '')
            
            # Use MLSystemV4's predict_race method
            v4_result = self.ml_system.predict_race(race_data, race_id)
            
            if not v4_result.get("success", False):
                return StandardizedResult.create_result(
                    race_id=race_id,
                    predictions=[],
                    metadata={
                        "adapter": "V4Adapter",
                        "method": "ml_system_error",
                        "original_error": v4_result.get("error", "Unknown error")
                    },
                    success=False,
                    error=v4_result.get("error", "MLSystemV4 prediction failed")
                )
            
            # Convert V4 predictions to standardized format
            standardized_predictions = []
            
            for pred in v4_result.get("predictions", []):
                standardized_pred = {
                    "dog": pred.get("dog_name", pred.get("dog", "Unknown")),
                    "raw_prob": pred.get("raw_win_probability", pred.get("win_probability", 0.0)),
                    "win_prob_norm": pred.get("normalized_win_probability", pred.get("win_probability", 0.0)),
                    "box_number": pred.get("box_number", 0),
                    "method": "ml_system_v4",
                    "confidence": pred.get("confidence"),
                    "place_probability": pred.get("place_probability"),
                    "expected_value": pred.get("expected_value"),
                    "calibration_applied": pred.get("calibration_applied", False)
                }
                standardized_predictions.append(standardized_pred)
            
            # V4 should already provide normalized probabilities, but ensure consistency
            if standardized_predictions and not any(pred.get("win_prob_norm", 0) > 0 for pred in standardized_predictions):
                # If no normalized probs, apply our normalization
                standardized_predictions = StandardizedResult.normalize_probabilities(standardized_predictions)
            
            # Sort by normalized probability
            standardized_predictions.sort(key=lambda x: x["win_prob_norm"], reverse=True)
            
            metadata = {
                "adapter": "V4Adapter",
                "method": "ml_system_v4",
                "model_info": v4_result.get("model_info", {}),
                "total_dogs": len(standardized_predictions),
                "temporal_leakage_protected": True,
                "calibration_applied": v4_result.get("calibration_applied", False),
                "ev_analysis": v4_result.get("ev_analysis", {})
            }
            
            return StandardizedResult.create_result(
                race_id=race_id,
                predictions=standardized_predictions,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ V4Adapter: Error predicting race {race_file_path}: {e}")
            return StandardizedResult.create_result(
                race_id=os.path.basename(race_file_path),
                predictions=[],
                metadata={"adapter": "V4Adapter", "method": "exception"},
                success=False,
                error=str(e)
            )
    
    def _preprocess_csv(self, race_file_path: str) -> pd.DataFrame:
        """
        Preprocess CSV file into format expected by MLSystemV4.
        
        Args:
            race_file_path: Path to the race CSV file
            
        Returns:
            Preprocessed DataFrame ready for MLSystemV4
        """
        try:
            # Read CSV file
            race_data = pd.read_csv(race_file_path)
            
            if race_data.empty:
                logger.warning(f"Empty CSV file: {race_file_path}")
                return pd.DataFrame()
            
            # Extract race information from filename
            filename = os.path.basename(race_file_path)
            race_parts = filename.replace('.csv', '').split(' - ')
            
            if len(race_parts) >= 3:
                venue = race_parts[1]
                date_part = race_parts[2]
                # Convert date format if needed
                try:
                    date_obj = datetime.strptime(date_part, "%d %B %Y")
                    race_date = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    race_date = datetime.now().strftime("%Y-%m-%d")
            else:
                venue = "Unknown"
                race_date = datetime.now().strftime("%Y-%m-%d")
            
            # Add race_date to DataFrame before preprocessing
            race_data['race_date'] = race_date
            race_data['venue'] = venue
            
            # Preprocess using MLSystemV4's method
            processed_data = self.ml_system.preprocess_upcoming_race_csv(
                race_data, 
                filename.replace('.csv', '')
            )
            
            logger.info(f"✅ V4Adapter: Preprocessed {len(processed_data)} dogs from {filename}")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ V4Adapter: Error preprocessing CSV {race_file_path}: {e}")
            return pd.DataFrame()


# Convenience functions for easy usage
def predict_with_v3(race_file_path: str, db_path: str = "greyhound_racing_data.db") -> Dict[str, Any]:
    """Convenience function to predict using V3Adapter."""
    adapter = V3Adapter(db_path)
    return adapter.predict_race(race_file_path)


def predict_with_v3s(race_file_path: str, db_path: str = "greyhound_racing_data.db") -> Dict[str, Any]:
    """Convenience function to predict using V3SAdapter."""
    adapter = V3SAdapter(db_path)
    return adapter.predict_race(race_file_path)


def predict_with_v4(race_file_path: str, db_path: str = "greyhound_racing_data.db") -> Dict[str, Any]:
    """Convenience function to predict using V4Adapter."""
    adapter = V4Adapter(db_path)
    return adapter.predict_race(race_file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prediction_adapters.py <race_file_path> [adapter_type]")
        print("Adapter types: v3, v3s, v4 (default: v4)")
        sys.exit(1)
    
    race_file = sys.argv[1]
    adapter_type = sys.argv[2].lower() if len(sys.argv) > 2 else "v4"
    
    print(f"🚀 Testing {adapter_type.upper()}Adapter with {race_file}")
    
    if adapter_type == "v3":
        result = predict_with_v3(race_file)
    elif adapter_type == "v3s":
        result = predict_with_v3s(race_file)
    elif adapter_type == "v4":
        result = predict_with_v4(race_file)
    else:
        print(f"❌ Unknown adapter type: {adapter_type}")
        sys.exit(1)
    
    print(f"📊 Result: {result}")
