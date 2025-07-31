#!/usr/bin/env python3
"""
Prediction Strategy Manager
===========================

Implements Strategy Pattern for prediction pipelines with async streaming and model loading optimization.

Key Features:
- Strategy Pattern: Primary = newest model, fallbacks registered dynamically
- Model loaded once per worker and reused
- Feature store reuse for efficiency  
- Parallel per-race predictions with concurrent.futures
- SSE streaming endpoint support

Author: AI Assistant
Date: July 31, 2025
"""

import os
import sys
import json
import logging
import asyncio
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from threading import Lock
from dataclasses import dataclass
import queue
import threading

# Import prediction components
from predictor import Predictor

try:
    from unified_predictor import UnifiedPredictor
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    
try:
    from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
    COMPREHENSIVE_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_AVAILABLE = False
    
try:
    from prediction_pipeline_v3 import PredictionPipelineV3
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False

# Feature store and model registry
try:
    from features.feature_store import FeatureStore
    from model_registry import get_model_registry
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structured prediction result"""
    race_id: str
    success: bool
    predictions: List[Dict[str, Any]]
    method_used: str
    timestamp: datetime
    processing_time_ms: float
    error_message: Optional[str] = None

@dataclass 
class StrategyConfig:
    """Configuration for prediction strategies"""
    strategy_name: str
    priority: int
    enabled: bool
    timeout_seconds: int = 300
    fallback_on_error: bool = True

class PredictionPipelineV3Strategy(Predictor):
    """Strategy implementation for PredictionPipelineV3"""
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """Ensure model is loaded once per worker"""
        if not self._model_loaded:
            with self.lock:
                if not self._model_loaded and V3_AVAILABLE:
                    try:
                        self.pipeline = PredictionPipelineV3()
                        self._model_loaded = True
                        logger.info("✅ PredictionPipelineV3 model loaded")
                    except Exception as e:
                        logger.error(f"❌ Failed to load PredictionPipelineV3: {e}")
                        
    def predict(self, race_file_path: str) -> PredictionResult:
        """Make prediction using PredictionPipelineV3"""
        start_time = time.time()
        
        try:
            self._ensure_model_loaded()
            if not self.pipeline:
                raise Exception("PredictionPipelineV3 not available")
                
            result = self.pipeline.predict_race_file(race_file_path, enhancement_level='full')
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResult(
                race_id=os.path.basename(race_file_path),
                success=result.get('success', False),
                predictions=result.get('predictions', []),
                method_used='PredictionPipelineV3',
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                error_message=result.get('error') if not result.get('success') else None
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"PredictionPipelineV3 strategy failed: {e}")
            return PredictionResult(
                race_id=os.path.basename(race_file_path),
                success=False,
                predictions=[],
                method_used='PredictionPipelineV3',
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                error_message=str(e)
            )

class ComprehensivePipelineStrategy(Predictor):
    """Strategy implementation for ComprehensivePredictionPipeline"""
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """Ensure model is loaded once per worker"""
        if not self._model_loaded:
            with self.lock:
                if not self._model_loaded and COMPREHENSIVE_AVAILABLE:
                    try:
                        self.pipeline = ComprehensivePredictionPipeline()
                        self._model_loaded = True
                        logger.info("✅ ComprehensivePredictionPipeline model loaded")
                    except Exception as e:
                        logger.error(f"❌ Failed to load ComprehensivePredictionPipeline: {e}")
                        
    def predict(self, race_file_path: str) -> PredictionResult:
        """Make prediction using ComprehensivePredictionPipeline"""
        start_time = time.time()
        
        try:
            self._ensure_model_loaded()
            if not self.pipeline:
                raise Exception("ComprehensivePredictionPipeline not available")
                
            result = self.pipeline.predict_race_file_with_all_enhancements(race_file_path)
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResult(
                race_id=os.path.basename(race_file_path),
                success=result.get('success', False),
                predictions=result.get('predictions', []),
                method_used='ComprehensivePredictionPipeline',
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                error_message=result.get('error') if not result.get('success') else None
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"ComprehensivePredictionPipeline strategy failed: {e}")
            return PredictionResult(
                race_id=os.path.basename(race_file_path),
                success=False,
                predictions=[],
                method_used='ComprehensivePredictionPipeline',
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                error_message=str(e)
            )

class UnifiedPredictorStrategy(Predictor):
    """Strategy implementation for UnifiedPredictor"""
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """Ensure model is loaded once per worker"""
        if not self._model_loaded:
            with self.lock:
                if not self._model_loaded and UNIFIED_AVAILABLE:
                    try:
                        self.predictor = UnifiedPredictor()
                        self._model_loaded = True
                        logger.info("✅ UnifiedPredictor model loaded")
                    except Exception as e:
                        logger.error(f"❌ Failed to load UnifiedPredictor: {e}")
                        
    def predict(self, race_file_path: str) -> PredictionResult:
        """Make prediction using UnifiedPredictor"""
        start_time = time.time()
        
        try:
            self._ensure_model_loaded()
            if not self.predictor:
                raise Exception("UnifiedPredictor not available")
                
            result = self.predictor.predict_race_file(race_file_path, enhancement_level='full')
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResult(
                race_id=os.path.basename(race_file_path),
                success=result.get('success', False),
                predictions=result.get('predictions', []),
                method_used='UnifiedPredictor',
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                error_message=result.get('error') if not result.get('success') else None
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"UnifiedPredictor strategy failed: {e}")
            return PredictionResult(
                race_id=os.path.basename(race_file_path),
                success=False,
                predictions=[],
                method_used='UnifiedPredictor',
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                error_message=str(e)
            )

class PredictionStrategyManager:
    """
    Manages prediction strategies with fallback hierarchy and parallel execution
    """
    
    def __init__(self):
        self.strategies: List[Predictor] = []
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.feature_store = None
        self.model_registry = None
        self._lock = Lock()
        self._stream_queues: Dict[str, queue.Queue] = {}
        
        # Initialize feature store and model registry if available
        if ADVANCED_FEATURES_AVAILABLE:
            try:
                self.feature_store = FeatureStore()
                self.model_registry = get_model_registry()
                logger.info("✅ Advanced features (FeatureStore, ModelRegistry) initialized")
            except Exception as e:
                logger.warning(f"⚠️ Advanced features initialization failed: {e}")
        
        self._register_strategies()
        logger.info(f"🚀 PredictionStrategyManager initialized with {len(self.strategies)} strategies")
    
    def _register_strategies(self):
        """Register available prediction strategies in priority order (newest first)"""
        
        # Strategy 1: PredictionPipelineV3 (newest, highest priority)
        if V3_AVAILABLE:
            self.strategies.append(PredictionPipelineV3Strategy())
            self.strategy_configs['PredictionPipelineV3'] = StrategyConfig(
                strategy_name='PredictionPipelineV3',
                priority=1,
                enabled=True,
                timeout_seconds=300
            )
            logger.info("✅ Registered PredictionPipelineV3Strategy (Priority 1)")
        
        # Strategy 2: ComprehensivePredictionPipeline (second priority)
        if COMPREHENSIVE_AVAILABLE:
            self.strategies.append(ComprehensivePipelineStrategy())
            self.strategy_configs['ComprehensivePredictionPipeline'] = StrategyConfig(
                strategy_name='ComprehensivePredictionPipeline',
                priority=2,
                enabled=True,
                timeout_seconds=300
            )
            logger.info("✅ Registered ComprehensivePipelineStrategy (Priority 2)")
        
        # Strategy 3: UnifiedPredictor (fallback)
        if UNIFIED_AVAILABLE:
            self.strategies.append(UnifiedPredictorStrategy())
            self.strategy_configs['UnifiedPredictor'] = StrategyConfig(
                strategy_name='UnifiedPredictor',
                priority=3,
                enabled=True,
                timeout_seconds=300
            )
            logger.info("✅ Registered UnifiedPredictorStrategy (Priority 3)")
    
    def predict_single_race(self, race_file_path: str) -> PredictionResult:
        """Predict a single race using the strategy pattern"""
        
        for strategy in self.strategies:
            config = self.strategy_configs.get(strategy.__class__.__name__.replace('Strategy', ''))
            if not config or not config.enabled:
                continue
                
            try:
                logger.info(f"🎯 Trying strategy: {config.strategy_name}")
                result = strategy.predict(race_file_path)
                
                if result.success:
                    logger.info(f"✅ Strategy {config.strategy_name} succeeded")
                    return result
                else:
                    logger.warning(f"⚠️ Strategy {config.strategy_name} failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Strategy {config.strategy_name} threw exception: {e}")
                continue
        
        # If all strategies failed
        return PredictionResult(
            race_id=os.path.basename(race_file_path),
            success=False,
            predictions=[],
            method_used='None (All strategies failed)',
            timestamp=datetime.now(),
            processing_time_ms=0,
            error_message="All prediction strategies failed"
        )
    
    def predict_races_parallel(self, race_file_paths: List[str], max_workers: int = 4) -> List[PredictionResult]:
        """Predict multiple races in parallel using concurrent.futures"""
        
        logger.info(f"🚀 Starting parallel prediction for {len(race_file_paths)} races with {max_workers} workers")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all prediction tasks
            future_to_race = {
                executor.submit(self.predict_single_race, race_path): race_path 
                for race_path in race_file_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_race):
                race_path = future_to_race[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✅" if result.success else "❌"
                    logger.info(f"{status} Completed prediction for {os.path.basename(race_path)} ({result.method_used})")
                except Exception as e:
                    logger.error(f"❌ Exception during prediction for {race_path}: {e}")
                    results.append(PredictionResult(
                        race_id=os.path.basename(race_path),
                        success=False,
                        predictions=[],
                        method_used='Exception',
                        timestamp=datetime.now(),
                        processing_time_ms=0,
                        error_message=str(e)
                    ))
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"🎉 Parallel prediction completed: {successful}/{len(results)} successful")
        
        return results
    
    def create_stream_queue(self, stream_id: str) -> queue.Queue:
        """Create a queue for streaming predictions"""
        with self._lock:
            stream_queue = queue.Queue()
            self._stream_queues[stream_id] = stream_queue
            return stream_queue
    
    def remove_stream_queue(self, stream_id: str):
        """Remove a streaming queue"""
        with self._lock:
            if stream_id in self._stream_queues:
                del self._stream_queues[stream_id]
    
    def predict_races_streaming(self, race_file_paths: List[str], stream_id: str, max_workers: int = 4):
        """Predict races with streaming updates"""
        
        stream_queue = self.create_stream_queue(stream_id)
        
        def worker_with_streaming(race_path):
            """Worker function that sends updates to the stream"""
            try:
                # Send start event
                stream_queue.put({
                    'type': 'start',
                    'race_id': os.path.basename(race_path),
                    'timestamp': datetime.now().isoformat()
                })
                
                result = self.predict_single_race(race_path)
                
                # Send result event
                stream_queue.put({
                    'type': 'result',
                    'race_id': result.race_id,
                    'success': result.success,
                    'predictions': result.predictions,
                    'method_used': result.method_used,
                    'processing_time_ms': result.processing_time_ms,
                    'error_message': result.error_message,
                    'timestamp': result.timestamp.isoformat()
                })
                
                return result
                
            except Exception as e:
                # Send error event
                stream_queue.put({
                    'type': 'error',
                    'race_id': os.path.basename(race_path),
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                raise
        
        # Start parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_with_streaming, race_path) for race_path in race_file_paths]
            
            # Wait for completion
            concurrent.futures.wait(futures)
            
            # Send completion event
            stream_queue.put({
                'type': 'complete',
                'total_races': len(race_file_paths),
                'timestamp': datetime.now().isoformat()
            })

# Global instance
_strategy_manager = None

def get_strategy_manager() -> PredictionStrategyManager:
    """Get global strategy manager instance"""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = PredictionStrategyManager()
    return _strategy_manager
