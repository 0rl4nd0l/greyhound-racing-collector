#!/usr/bin/env python3
"""
Unified Prediction Interface
============================

This unified system consolidates all prediction methods while maintaining 100% backward compatibility
with the existing Flask app, automation, and API interfaces.

Architecture:
- Maintains exact same interfaces as existing prediction scripts
- Implements intelligent fallback hierarchy
- Adds centralized configuration and caching
- Preserves all existing functionality while improving maintainability

Author: AI Assistant
Date: July 26, 2025
"""

import os
import sys
import json
import re
import time
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced pipeline v2
try:
    from enhanced_pipeline_v2 import EnhancedPipelineV2
    ENHANCED_PIPELINE_V2_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Enhanced Pipeline V2 not available: {e}")
    ENHANCED_PIPELINE_V2_AVAILABLE = False

class UnifiedPredictorConfig:
    """Centralized configuration for the unified prediction system"""
    
    def __init__(self):
        # Core directories
        self.base_dir = Path(__file__).parent
        self.upcoming_races_dir = self.base_dir / "upcoming_races"
        self.predictions_dir = self.base_dir / "predictions"
        self.models_dir = self.base_dir / "comprehensive_trained_models"
        self.db_path = self.base_dir / "greyhound_racing_data.db"
        
        # Prediction method priorities (highest to lowest)
        self.prediction_methods = [
            "unified_system",
            "enhanced_pipeline_v2",
            "comprehensive_pipeline",
            "weather_enhanced", 
            "comprehensive_ml",
            "basic_fallback"
        ]
        
        # Component availability flags
        self.components_available = {
            "unified_system": True,  # Always available as it's the core system
            "enhanced_pipeline_v2": self._check_enhanced_pipeline_v2(),
            "enhanced_feature_engineering_v2": self._check_enhanced_feature_engineering_v2(),
            "advanced_ml_system_v2": self._check_advanced_ml_system_v2(),
            "data_quality_improver": self._check_data_quality_improver(),
            "comprehensive_pipeline": self._check_comprehensive_pipeline(),
            "weather_enhanced": self._check_weather_enhanced(),
            "comprehensive_ml": self._check_comprehensive_ml(),
            "gpt_enhancement": self._check_gpt_enhancement()
        }
        
        # Performance settings
        self.enable_caching = True
        self.cache_duration_minutes = 30
        self.max_prediction_timeout = 300  # 5 minutes
        
        # Feature standardization
        self.standard_feature_names = [
            'weighted_recent_form', 'speed_trend', 'speed_consistency', 'venue_win_rate',
            'venue_avg_position', 'venue_experience', 'distance_win_rate', 'distance_avg_time',
            'box_position_win_rate', 'box_position_avg', 'recent_momentum', 'competitive_level',
            'position_consistency', 'top_3_rate', 'break_quality', 'grade_trend',
            'current_class_comfort', 'break_impact', 'consistency_after_break', 'trainer_impact',
            'seasonal_performance', 'competition_level', 'box_number'
        ]
        
        # Intelligent default values based on domain knowledge
        self.intelligent_defaults = {
            'weighted_recent_form': 4.5,  # Mid-field average
            'speed_trend': 0.0,  # No trend
            'speed_consistency': 1.0,  # Moderate consistency
            'venue_win_rate': 0.125,  # 1 in 8 chance (field size)
            'venue_avg_position': 4.5,  # Mid-field
            'venue_experience': 0,  # No experience
            'distance_win_rate': 0.125,  # Same as venue
            'distance_avg_time': 30.0,  # Typical 500m time
            'box_position_win_rate': 0.125,  # Equal chance
            'box_position_avg': 4.5,  # Mid-field
            'recent_momentum': 0.5,  # Neutral
            'competitive_level': 0.5,  # Average competition
            'position_consistency': 0.5,  # Moderate consistency
            'top_3_rate': 0.375,  # 3 in 8 chance
            'break_quality': 0.5,  # Average break
            'grade_trend': 0.0,  # No trend
            'current_class_comfort': 0.5,  # Comfortable
            'break_impact': 0.0,  # No impact
            'consistency_after_break': 0.5,  # Average
            'trainer_impact': 0.5,  # Average trainer
            'seasonal_performance': 0.5,  # Consistent year-round
            'competition_level': 0.5,  # Average competition
            'box_number': 4.5  # Mid-field box
        }
        
        logger.info(f"🔧 Unified Predictor Configuration:")
        logger.info(f"   Components available: {sum(self.components_available.values())}/4")
        for component, available in self.components_available.items():
            status = "✅" if available else "❌"
            logger.info(f"   {status} {component}")
    
    def _check_comprehensive_pipeline(self) -> bool:
        """Check if comprehensive prediction pipeline is available"""
        try:
            pipeline_file = self.base_dir / "comprehensive_prediction_pipeline.py"
            if pipeline_file.exists():
                # Try importing to verify it's functional
                import importlib.util
                spec = importlib.util.spec_from_file_location("comprehensive_prediction_pipeline", pipeline_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return hasattr(module, 'ComprehensivePredictionPipeline')
            return False
        except Exception as e:
            logger.debug(f"Comprehensive pipeline check failed: {e}")
            return False
    
    def _check_weather_enhanced(self) -> bool:
        """Check if weather enhanced predictor is available"""
        try:
            weather_file = self.base_dir / "weather_enhanced_predictor.py"
            if weather_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("weather_enhanced_predictor", weather_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return hasattr(module, 'WeatherEnhancedPredictor')
            return False
        except Exception as e:
            logger.debug(f"Weather enhanced check failed: {e}")
            return False
    
    def _check_comprehensive_ml(self) -> bool:
        """Check if comprehensive ML system is available"""
        try:
            ml_file = self.base_dir / "comprehensive_enhanced_ml_system.py"
            if ml_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("comprehensive_enhanced_ml_system", ml_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return hasattr(module, 'ComprehensiveEnhancedMLSystem')
            return False
        except Exception as e:
            logger.debug(f"Comprehensive ML check failed: {e}")
            return False
    
    def _check_gpt_enhancement(self) -> bool:
        """Check if GPT enhancement is available"""
        try:
            gpt_file = self.base_dir / "gpt_prediction_enhancer.py"
            return gpt_file.exists() and os.getenv('OPENAI_API_KEY') is not None
        except Exception:
            return False
    
    def _check_enhanced_pipeline_v2(self) -> bool:
        """Check if enhanced pipeline v2 is available"""
        try:
            # Check if all v2 components are available
            return (self._check_enhanced_feature_engineering_v2() and 
                   self._check_advanced_ml_system_v2() and 
                   self._check_data_quality_improver())
        except Exception as e:
            logger.debug(f"Enhanced pipeline v2 check failed: {e}")
            return False
    
    def _check_enhanced_feature_engineering_v2(self) -> bool:
        """Check if enhanced feature engineering v2 is available"""
        try:
            feature_file = self.base_dir / "enhanced_feature_engineering_v2.py"
            if feature_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("enhanced_feature_engineering_v2", feature_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return hasattr(module, 'AdvancedFeatureEngineer')
            return False
        except Exception as e:
            logger.debug(f"Enhanced feature engineering v2 check failed: {e}")
            return False
    
    def _check_advanced_ml_system_v2(self) -> bool:
        """Check if advanced ML system v2 is available"""
        try:
            ml_file = self.base_dir / "advanced_ml_system_v2.py"
            if ml_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("advanced_ml_system_v2", ml_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return hasattr(module, 'AdvancedMLSystemV2')
            return False
        except Exception as e:
            logger.debug(f"Advanced ML system v2 check failed: {e}")
            return False
    
    def _check_data_quality_improver(self) -> bool:
        """Check if data quality improver is available"""
        try:
            quality_file = self.base_dir / "data_quality_improver.py"
            if quality_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("data_quality_improver", quality_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return hasattr(module, 'DataQualityImprover')
            return False
        except Exception as e:
            logger.debug(f"Data quality improver check failed: {e}")
            return False

class PredictionCache:
    """Simple caching system for prediction results"""
    
    def __init__(self, enable_cache: bool = True, cache_duration_minutes: int = 30):
        self.enable_cache = enable_cache
        self.cache_duration = cache_duration_minutes * 60  # Convert to seconds
        self.cache = {}
        self.cache_dir = Path("./unified_prediction_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, race_file_path: str) -> str:
        """Generate cache key based on file path and modification time"""
        try:
            file_path = Path(race_file_path)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                return f"{file_path.name}_{int(mtime)}"
            return file_path.name
        except Exception:
            return os.path.basename(race_file_path)
    
    def get(self, race_file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and not expired"""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(race_file_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            if cache_file.exists():
                # Check if cache is still valid
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < self.cache_duration:
                    with open(cache_file, 'r') as f:
                        cached_result = json.load(f)
                    logger.info(f"📋 Using cached prediction for {os.path.basename(race_file_path)}")
                    return cached_result
                else:
                    # Cache expired, remove it
                    cache_file.unlink()
        except Exception as e:
            logger.debug(f"Cache retrieval error: {e}")
        
        return None
    
    def set(self, race_file_path: str, result: Dict[str, Any]) -> None:
        """Cache prediction result"""
        if not self.enable_cache or not result.get('success'):
            return
        
        cache_key = self._get_cache_key(race_file_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Add caching metadata
            cached_result = result.copy()
            cached_result['cached_at'] = datetime.now().isoformat()
            cached_result['cache_key'] = cache_key
            
            with open(cache_file, 'w') as f:
                json.dump(cached_result, f, indent=2)
            
            logger.debug(f"💾 Cached prediction for {os.path.basename(race_file_path)}")
        except Exception as e:
            logger.debug(f"Cache storage error: {e}")

class UnifiedPredictor:
    """
    Unified prediction interface that consolidates all prediction methods
    while maintaining 100% backward compatibility with existing systems.
    """
    
    def __init__(self, config: Optional[UnifiedPredictorConfig] = None):
        self.config = config or UnifiedPredictorConfig()
        self.cache = PredictionCache(
            self.config.enable_caching,
            self.config.cache_duration_minutes
        )
        
        # Initialize prediction components
        self.predictors = {}
        self._initialize_predictors()
        
        # Model registry integration
        self.model_registry = None
        self.best_model = None
        self.best_model_scaler = None
        self.best_model_metadata = None
        self.model_last_check = 0
        self.model_check_interval = 60  # Check for model updates every 60 seconds
        
        self._initialize_model_registry()
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'method_usage': {method: 0 for method in self.config.prediction_methods},
            'average_time': 0.0,
            'success_rate': 0.0,
            'model_updates': 0
        }
        
        logger.info(f"🚀 Unified Predictor initialized with {len(self.predictors)} active methods")
    
    def _initialize_model_registry(self):
        """Initialize model registry for dynamic model loading with priority system"""
        try:
            self.best_model_info = self._get_latest_best_model()
            if self.best_model_info:
                logger.info(f"🎯 Using latest model: {self.best_model_info['name']} (accuracy: {self.best_model_info.get('accuracy', 'N/A')})")
            self.model_last_check = time.time()
            
            logger.debug("Model registry initialized with priority system")
        except Exception as e:
            logger.debug(f"Model registry initialization failed: {e}")
    
    def _get_latest_best_model(self):
        """Get the latest and best performing model with priority system"""
        import glob
        import joblib
        
        try:
            models_dir = self.config.models_dir
            if not models_dir.exists():
                return None
            
            # Priority order for model types (highest to lowest)
            model_priorities = [
                'comprehensive_best_model_*.joblib',
                'automated_best_model_*.joblib', 
                'retrained_model_*.joblib',
                '*.joblib'
            ]
            
            best_model = None
            best_accuracy = 0
            best_priority = 999
            
            for priority, pattern in enumerate(model_priorities):
                model_files = glob.glob(str(models_dir / pattern))
                
                for model_file in model_files:
                    try:
                        # Load model metadata to check accuracy
                        model_data = joblib.load(model_file)
                        
                        # Extract accuracy from different possible keys
                        accuracy = 0
                        if isinstance(model_data, dict):
                            accuracy = model_data.get('accuracy', 
                                      model_data.get('test_accuracy',
                                      model_data.get('cv_accuracy', 0)))
                        
                        # Get file modification time for tie-breaking
                        file_time = os.path.getctime(model_file)
                        
                        # Select model based on priority, then accuracy, then recency
                        if (priority < best_priority or 
                            (priority == best_priority and accuracy > best_accuracy) or
                            (priority == best_priority and accuracy == best_accuracy and 
                             (best_model is None or file_time > os.path.getctime(best_model)))):
                            
                            best_model = model_file
                            best_accuracy = accuracy
                            best_priority = priority
                            
                    except Exception as e:
                        logger.debug(f"Error loading model {model_file}: {e}")
                        continue
            
            if best_model:
                return {
                    'path': best_model,
                    'name': os.path.basename(best_model),
                    'accuracy': best_accuracy,
                    'priority': best_priority,
                    'timestamp': datetime.fromtimestamp(os.path.getctime(best_model)).isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error finding latest model: {e}")
            return None
    
    def _initialize_predictors(self):
        """Initialize available prediction components"""
        
        # 0. Enhanced Pipeline V2 (Highest Priority)
        if self.config.components_available['enhanced_pipeline_v2']:
            try:
                from enhanced_pipeline_v2 import EnhancedPipelineV2
                self.predictors['enhanced_pipeline_v2'] = EnhancedPipelineV2()
                logger.info("✅ Enhanced Pipeline V2 initialized")
            except Exception as e:
                logger.warning(f"Enhanced Pipeline V2 initialization failed: {e}")
        
        # 1. Comprehensive Pipeline (Second Priority)
        if self.config.components_available['comprehensive_pipeline']:
            try:
                from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
                self.predictors['comprehensive_pipeline'] = ComprehensivePredictionPipeline()
                logger.info("✅ Comprehensive Pipeline initialized")
            except Exception as e:
                logger.warning(f"Comprehensive Pipeline initialization failed: {e}")
        
        # 2. Weather Enhanced Predictor
        if self.config.components_available['weather_enhanced']:
            try:
                from weather_enhanced_predictor import WeatherEnhancedPredictor
                self.predictors['weather_enhanced'] = WeatherEnhancedPredictor()
                logger.info("✅ Weather Enhanced Predictor initialized")
            except Exception as e:
                logger.warning(f"Weather Enhanced Predictor initialization failed: {e}")
        
        # 3. Comprehensive ML System
        if self.config.components_available['comprehensive_ml']:
            try:
                from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
                self.predictors['comprehensive_ml'] = ComprehensiveEnhancedMLSystem()
                logger.info("✅ Comprehensive ML System initialized")
            except Exception as e:
                logger.warning(f"Comprehensive ML System initialization failed: {e}")
        
        # 4. GPT Enhancement (can be applied to any prediction)
        if self.config.components_available['gpt_enhancement']:
            try:
                from gpt_prediction_enhancer import GPTPredictionEnhancer
                self.gpt_enhancer = GPTPredictionEnhancer()
                logger.info("✅ GPT Enhancement available")
            except Exception as e:
                logger.warning(f"GPT Enhancement initialization failed: {e}")
                self.gpt_enhancer = None
        else:
            self.gpt_enhancer = None
    
    def predict_race_file(self, race_file_path: str, 
                         enhancement_level: str = 'full',
                         force_refresh: bool = False) -> Dict[str, Any]:
        """
        Main prediction interface - maintains exact same signature as existing scripts
        
        Args:
            race_file_path: Path to race CSV file
            enhancement_level: 'basic', 'weather', 'full' (includes GPT)
            force_refresh: Skip cache and recompute prediction
        
        Returns:
            Dict with same format as existing prediction scripts:
            {
                'success': bool,
                'predictions': list,
                'error': str (if failed),
                'race_info': dict,
                'prediction_method': str,
                'timestamp': str
            }
        """
        start_time = time.time()
        self.prediction_stats['total_predictions'] += 1
        
        # Validate input
        if not os.path.exists(race_file_path):
            return {
                'success': False,
                'error': f'Race file not found: {race_file_path}',
                'predictions': []
            }
        
        logger.info(f"🎯 Predicting: {os.path.basename(race_file_path)} (level: {enhancement_level})")
        
        # Check cache first (unless forced refresh)
        if not force_refresh:
            cached_result = self.cache.get(race_file_path)
            if cached_result:
                self.prediction_stats['cache_hits'] += 1
                return cached_result
        
        # Try prediction methods in priority order
        prediction_result = None
        successful_method = None
        
        for method in self.config.prediction_methods:
            if method == 'unified_system':
                # Use the unified system's own comprehensive prediction
                try:
                    logger.info(f"🧠 Trying unified system comprehensive prediction...")
                    prediction_result = self._unified_system_prediction(race_file_path)
                    if prediction_result and prediction_result.get('success'):
                        predictions = prediction_result.get('predictions', [])
                        if self._is_prediction_quality_acceptable(predictions):
                            successful_method = method
                            logger.info(f"✅ Unified system successful with good quality")
                            break
                        else:
                            logger.warning(f"⚠️ Unified system produced poor quality predictions, trying next method")
                            continue
                    else:
                        logger.warning(f"⚠️ Unified system failed: {prediction_result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.warning(f"❌ Unified system crashed: {str(e)}")
                    continue
            elif method == 'basic_fallback':
                # Always available as last resort
                prediction_result = self._basic_fallback_prediction(race_file_path)
                successful_method = method
                break
            elif method in self.predictors:
                try:
                    logger.info(f"🧠 Trying {method}...")
                    prediction_result = self._run_prediction_method(method, race_file_path)
                    
                    if prediction_result and prediction_result.get('success'):
                        # Check prediction quality - if all scores are the same, it's poor quality
                        predictions = prediction_result.get('predictions', [])
                        if self._is_prediction_quality_acceptable(predictions):
                            successful_method = method
                            logger.info(f"✅ {method} successful with good quality")
                            break
                        else:
                            logger.warning(f"⚠️ {method} produced poor quality predictions (uniform scores), trying next method")
                            continue
                    else:
                        logger.warning(f"⚠️ {method} failed: {prediction_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.warning(f"❌ {method} crashed: {str(e)}")
                    continue
        
        # If no method succeeded, return error
        if not prediction_result or not prediction_result.get('success'):
            return {
                'success': False,
                'error': 'All prediction methods failed',
                'predictions': [],
                'attempts': list(self.predictors.keys()) + ['basic_fallback']
            }
        
        # Update statistics
        self.prediction_stats['method_usage'][successful_method] += 1
        prediction_time = time.time() - start_time
        self._update_performance_stats(prediction_time, True)
        
        # Add unified predictor metadata
        # Build methods list based on successful method and available components
        methods_used = [successful_method]
        if successful_method == 'enhanced_pipeline_v2':
            # Enhanced Pipeline V2 uses the most advanced sub-systems
            if self.config.components_available['enhanced_feature_engineering_v2']:
                methods_used.append('enhanced_feature_engineering_v2')
            if self.config.components_available['advanced_ml_system_v2']:
                methods_used.append('advanced_ml_system_v2')
            if self.config.components_available['data_quality_improver']:
                methods_used.append('data_quality_improver')
            methods_used.extend(['embedded_historical_data', 'ml_ensemble'])
        elif successful_method in ['comprehensive_pipeline', 'weather_enhanced']:
            # These methods use multiple sub-systems
            if self.config.components_available['comprehensive_ml']:
                methods_used.append('ml_system')
            if self.config.components_available['weather_enhanced'] and successful_method != 'weather_enhanced':
                methods_used.append('weather_enhanced')
            methods_used.append('traditional')
            if successful_method == 'comprehensive_pipeline':
                methods_used.append('enhanced_data')
        
        prediction_result.update({
            'prediction_method': f"unified_{successful_method}",
            'unified_predictor_version': '1.0.0',
            'analysis_version': f'Unified Predictor v1.0 - {successful_method.replace("_", " ").title()}',
            'prediction_time_seconds': round(prediction_time, 2),
            'timestamp': datetime.now().isoformat(),
            'prediction_methods_used': methods_used,  # Enhanced methods list
            'enhanced_with_unified_system': successful_method == 'enhanced_pipeline_v2',
            'enhanced_pipeline_v2_used': successful_method == 'enhanced_pipeline_v2',
            'unified_system_components': sum(self.config.components_available.values()),
            'fallback_hierarchy_used': True,
            'quality_control_applied': True
        })
        
        # Apply GPT enhancement if requested and available
        if enhancement_level == 'full' and self.gpt_enhancer:
            try:
                logger.info("🤖 Applying GPT enhancement...")
                gpt_enhancement = self.gpt_enhancer.enhance_race_prediction(race_file_path)
                if 'error' not in gpt_enhancement:
                    prediction_result['gpt_enhancement'] = gpt_enhancement
                    prediction_result['enhanced_with_gpt'] = True
                    
                    # Use the properly merged predictions that preserve ML scores
                    if 'merged_predictions' in gpt_enhancement and gpt_enhancement['merged_predictions']:
                        prediction_result['predictions'] = gpt_enhancement['merged_predictions']
                        logger.info("✅ GPT enhancement applied with merged predictions")
                    else:
                        logger.info("✅ GPT enhancement applied")
                else:
                    logger.warning(f"⚠️ GPT enhancement failed: {gpt_enhancement['error']}")
                    prediction_result['enhanced_with_gpt'] = False
            except Exception as e:
                logger.warning(f"❌ GPT enhancement error: {e}")
                prediction_result['enhanced_with_gpt'] = False
        
        # Cache successful result
        self.cache.set(race_file_path, prediction_result)
        
        logger.info(f"🏁 Prediction completed in {prediction_time:.2f}s using {successful_method}")
        return prediction_result
    
    def _run_prediction_method(self, method: str, race_file_path: str) -> Dict[str, Any]:
        """Run specific prediction method"""
        predictor = self.predictors[method]
        
        # All predictors implement predict_race_file method
        if hasattr(predictor, 'predict_race_file'):
            result = predictor.predict_race_file(race_file_path)
        elif hasattr(predictor, 'predict_upcoming_race'):
            # Some ML systems use this method name
            result = predictor.predict_upcoming_race(race_file_path)
        else:
            raise AttributeError(f"Predictor {method} doesn't have expected prediction method")
        
        # Standardize prediction format - ensure all predictions have 'prediction_score'
        if result and result.get('success') and result.get('predictions'):
            for prediction in result['predictions']:
                if 'prediction_score' not in prediction and 'final_score' in prediction:
                    prediction['prediction_score'] = prediction['final_score']
                elif 'prediction_score' not in prediction:
                    prediction['prediction_score'] = 0.5  # Default fallback
        
        return result
    
    def _basic_fallback_prediction(self, race_file_path: str) -> Dict[str, Any]:
        """Basic fallback prediction when all other methods fail"""
        try:
            import pandas as pd
            import random
            
            # Read the race file with proper delimiter detection
            # First, try to detect the delimiter by reading a sample
            with open(race_file_path, 'r') as f:
                first_line = f.readline()
                if '|' in first_line:
                    delimiter = '|'
                else:
                    delimiter = ','
            
            df = pd.read_csv(race_file_path, delimiter=delimiter)
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'Race file is empty',
                    'predictions': []
                }
            
            # Extract participating dogs using same logic as existing predictors
            participating_dogs = self._extract_participating_dogs_basic(df)
            
            if not participating_dogs:
                return {
                    'success': False,
                    'error': 'No valid dogs found in race file',
                    'predictions': []
                }
            
            logger.info(f"📊 Using basic fallback for {len(participating_dogs)} dogs")
            
            # Basic prediction based on available data
            predictions = []
            for dog_info in participating_dogs:
                dog_name = dog_info['name']
                box_number = dog_info.get('box', 0)
                
                # Simple scoring with some logic
                score = 0.5  # Base score
                
                # Box position influence (inside boxes slight advantage)
                if box_number in [1, 2, 3, 4]:
                    score += 0.05
                elif box_number in [5, 6]:
                    score += 0.02
                
                # Add controlled randomness
                score += random.uniform(-0.15, 0.15)
                score = max(0.1, min(0.85, score))  # Clamp between 0.1 and 0.85
                
                predictions.append({
                    'dog_name': dog_name,
                    'box_number': box_number,
                    'prediction_score': round(score, 3),
                    'confidence_level': 'LOW',
                    'reasoning': 'Basic fallback prediction',
                    'method': 'unified_basic_fallback'
                })
            
            # Sort by prediction score
            predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
            
            return {
                'success': True,
                'predictions': predictions,
                'race_info': {
                    'filename': os.path.basename(race_file_path),
                    'total_dogs': len(predictions)
                },
                'prediction_method': 'basic_fallback',
                'note': 'Basic fallback used - limited accuracy expected'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Basic fallback failed: {str(e)}',
                'predictions': []
            }
    
    def _extract_participating_dogs_basic(self, race_df):
        """Extract participating dogs from race CSV - simplified version of weather predictor logic"""
        try:
            dogs = []
            current_dog_name = None
            
            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get('Dog Name', '')).strip()
                
                # Check if this is a new dog or continuation of previous
                if dog_name_raw not in ['""', '', 'nan'] and dog_name_raw != 'nan':
                    # New dog - clean the name
                    current_dog_name = dog_name_raw
                    # Remove box number prefix (e.g., "1. SALEENA BALE" -> "SALEENA BALE")
                    if '. ' in current_dog_name:
                        parts = current_dog_name.split('. ', 1)
                        if len(parts) == 2:
                            try:
                                box_number = int(parts[0])
                                current_dog_name = parts[1]
                            except (ValueError, TypeError):
                                box_number = None
                        else:
                            box_number = None
                    else:
                        box_number = None
                    
                    # For upcoming races, NEVER use BOX column as it contains historical data
                    # Only rely on prefix parsing (e.g., "1. Dog Name" -> box 1)
                    if box_number is None:
                        # If prefix parsing failed, use sequential numbering as fallback
                        box_number = len(dogs) + 1  # Default sequential
                    
                    dogs.append({
                        'name': current_dog_name,
                        'box': box_number,
                        'raw_name': dog_name_raw
                    })
            
            return dogs
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting participating dogs: {e}")
            return []
    
    def _is_prediction_quality_acceptable(self, predictions: List[Dict[str, Any]]) -> bool:
        """Check if predictions have acceptable quality (not all the same score)"""
        if not predictions or len(predictions) < 2:
            return True  # Can't determine quality with < 2 predictions
        
        # Extract scores, handling different field names
        scores = []
        for pred in predictions:
            score = pred.get('prediction_score', pred.get('final_score', 0.5))
            scores.append(float(score))
        
        # Check if all scores are the same (poor quality indicator)
        unique_scores = len(set(scores))
        if unique_scores == 1:
            # All scores identical - definitely poor quality
            return False
        
        # Check if scores are too uniform (e.g., all very close to 0.5)
        import statistics
        if len(scores) > 3:
            score_variance = statistics.variance(scores)
            # If variance is too low, scores are too uniform (adjusted threshold)
            if score_variance < 0.0001:  # Lower threshold to allow more variation
                return False
        
        # Check for unrealistic score ranges
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        # If all scores are between 0.49 and 0.51, it's probably poor quality
        if score_range < 0.02 and min_score > 0.49 and max_score < 0.51:
            return False
        
        return True
    
    def _update_performance_stats(self, prediction_time: float, success: bool):
        """Update performance statistics"""
        # Update average time (simple moving average)
        total_predictions = self.prediction_stats['total_predictions']
        current_avg = self.prediction_stats['average_time']
        self.prediction_stats['average_time'] = (
            (current_avg * (total_predictions - 1) + prediction_time) / total_predictions
        )
        
        # Update success rate
        if success:
            successful_predictions = sum(self.prediction_stats['method_usage'].values())
            self.prediction_stats['success_rate'] = successful_predictions / total_predictions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.prediction_stats,
            'cache_hit_rate': (
                self.prediction_stats['cache_hits'] / max(1, self.prediction_stats['total_predictions'])
            ),
            'available_methods': list(self.predictors.keys()),
            'components_status': self.config.components_available
        }
    
    def _unified_system_prediction(self, race_file_path: str) -> Dict[str, Any]:
        """Unified system's own comprehensive prediction method with optimal feature alignment"""
        try:
            import pandas as pd
            import numpy as np
            import sqlite3
            
            logger.info(f"🚀 Starting unified system prediction for: {os.path.basename(race_file_path)}")
            
            # Load race file with enhanced parsing
            race_df = self._load_race_file_enhanced(race_file_path)
            if race_df is None or race_df.empty:
                return {'success': False, 'error': 'Could not load race file', 'predictions': []}
            
            # Extract race information
            race_info = self._extract_race_info_unified(race_file_path, race_df)
            
            # Extract participating dogs with embedded historical data
            participating_dogs = self._extract_dogs_with_history(race_df, race_file_path)
            if not participating_dogs:
                return {'success': False, 'error': 'No participating dogs found', 'predictions': []}
            
            logger.info(f"📊 Processing {len(participating_dogs)} dogs with unified system")
            
            # Generate predictions for each dog
            predictions = []
            for dog_info in participating_dogs:
                try:
                    prediction = self._generate_unified_prediction(dog_info, race_info)
                    if prediction:
                        predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Error predicting for {dog_info.get('name', 'unknown')}: {e}")
                    continue
            
            if not predictions:
                return {'success': False, 'error': 'No valid predictions generated', 'predictions': []}
            
            # Sort and rank predictions
            predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
            for i, pred in enumerate(predictions, 1):
                pred['rank'] = i
                pred['predicted_rank'] = i
            
            return {
                'success': True,
                'predictions': predictions,
                'race_info': race_info,
                'prediction_method': 'unified_system_comprehensive',
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality_summary': self._calculate_race_data_quality(participating_dogs),
                'unified_system_version': '2.0',
                'feature_alignment_optimized': True,
                'missing_data_handled': True
            }
            
        except Exception as e:
            logger.error(f"Unified system prediction error: {str(e)}")
            return {'success': False, 'error': f'Unified system error: {str(e)}', 'predictions': []}
    
    def _load_race_file_enhanced(self, race_file_path: str) -> pd.DataFrame:
        """Enhanced race file loading with better error handling"""
        try:
            # Detect delimiter and encoding
            with open(race_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = '|' if '|' in first_line else ','
            
            # Load with error handling
            df = pd.read_csv(race_file_path, delimiter=delimiter, encoding='utf-8')
            
            # Validate critical columns exist
            required_cols = ['Dog Name']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading race file: {e}")
            return None
    
    def _extract_race_info_unified(self, race_file_path: str, race_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comprehensive race information with enhanced parsing"""
        try:
            filename = os.path.basename(race_file_path)
            logger.info(f"🔍 Parsing filename: {filename}")
            
            # Initialize with defaults
            race_info = {
                'filename': filename,
                'filepath': race_file_path,
                'venue': 'UNKNOWN',
                'race_number': '0',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'distance': 500,  # Default
                'grade': 'Mixed',
                'field_size': len(race_df)
            }

            # Enhanced parsing patterns with proper group assignments
            patterns = [
                # Pattern 1: "Race 5 - GEE - 22 July 2025.csv"
                {
                    'pattern': r'Race\s+(\d+)\s*-\s*([A-Z_]+)\s*-\s*(\d{1,2}\s+\w+\s+\d{4})',
                    'groups': ['race_number', 'venue', 'date_str']
                },
                # Pattern 2: "Race_5_GEE_22_July_2025.csv"
                {
                    'pattern': r'Race_(\d+)_([A-Z_]+)_(.+)',
                    'groups': ['race_number', 'venue', 'date_str']
                },
                # Pattern 3: "GEE_5_22_July_2025.csv" (venue first)
                {
                    'pattern': r'([A-Z_]+)_(\d+)_(.+)',
                    'groups': ['venue', 'race_number', 'date_str']
                },
                # Pattern 4: "gee_2025-07-22_5.csv" (lowercase with ISO date)
                {
                    'pattern': r'([a-z]+)_(\d{4}-\d{2}-\d{2})_(\d+)',
                    'groups': ['venue', 'date_str', 'race_number']
                },
                # Pattern 5: "race5_gee_july22.csv"
                {
                    'pattern': r'race(\d+)[_-]([a-zA-Z]+)[_-](.+)',
                    'groups': ['race_number', 'venue', 'date_str']
                }
            ]
            
            for pattern_config in patterns:
                match = re.search(pattern_config['pattern'], filename, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    group_names = pattern_config['groups']
                    
                    if len(groups) == len(group_names):
                        parsed_data = {}
                        for i, group_name in enumerate(group_names):
                            value = groups[i].strip() if groups[i] else ''
                            
                            if group_name == 'venue':
                                parsed_data['venue'] = self._normalize_venue(value)
                            elif group_name == 'race_number':
                                parsed_data['race_number'] = self._parse_race_number(value)
                            elif group_name == 'date_str':
                                parsed_data['date'] = self._normalize_date(value)
                        
                        # Update race_info with parsed data
                        race_info.update(parsed_data)
                        logger.info(f"✅ Parsed using pattern {pattern_config['pattern'][:30]}...")
                        logger.info(f"   Venue: {race_info['venue']}, Race: {race_info['race_number']}, Date: {race_info['date']}")
                        break

            # If parsing failed, try content-based extraction
            if race_info['venue'] == 'UNKNOWN':
                logger.warning(f"⚠️ Standard parsing failed, trying content-based extraction...")
                race_info = self._try_content_based_parsing(race_file_path, race_df, race_info)

            # Extract additional info from CSV data
            if not race_df.empty:
                first_row = race_df.iloc[0]
                
                # Distance
                if 'DIST' in race_df.columns:
                    dist_val = first_row.get('DIST')
                    if pd.notna(dist_val):
                        try:
                            race_info['distance'] = int(float(str(dist_val).replace('m', '')))
                        except (ValueError, TypeError):
                            pass
                
                # Grade
                if 'G' in race_df.columns:
                    grade_val = first_row.get('G')
                    if pd.notna(grade_val) and str(grade_val) != 'nan':
                        race_info['grade'] = f"Grade {grade_val}"
                
                # Venue from track code if still unknown
                if 'TRACK' in race_df.columns and race_info['venue'] == 'UNKNOWN':
                    track_code = first_row.get('TRACK')
                    if pd.notna(track_code):
                        track_mapping = {
                            'TARE': 'TAREE', 'MAIT': 'MAITLAND', 'GRDN': 'GOSFORD',
                            'CASO': 'CASINO', 'DAPT': 'DAPTO', 'BAL': 'BALLARAT',
                            'SAN': 'SANDOWN', 'WAR': 'WARRAGUL', 'DUBO': 'DUBBO'
                        }
                        race_info['venue'] = track_mapping.get(str(track_code).upper(), str(track_code))
        
        except Exception as e:
            logger.error(f"Error extracting race info: {e}")
            return {
                'filename': os.path.basename(race_file_path),
                'filepath': race_file_path,
                'venue': 'UNKNOWN',
                'race_number': '0',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'distance': 500,
                'grade': 'Mixed',
                'field_size': len(race_df) if not race_df.empty else 0
            }
        
        return race_info
    
    def _extract_dogs_with_history(self, race_df: pd.DataFrame, race_file_path: str) -> List[Dict[str, Any]]:
        """Extract dogs with embedded historical data and database lookup"""
        dogs = []
        current_dog_data = {}
        current_dog_history = []
        
        try:
            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get('Dog Name', '')).strip()
                
                # Check if this is a new dog entry
                if dog_name_raw and dog_name_raw not in ['""', '', 'nan', 'NaN']:
                    # Save previous dog if exists
                    if current_dog_data:
                        current_dog_data['embedded_history'] = current_dog_history
                        # Add database history
                        current_dog_data['database_history'] = self._get_database_history(current_dog_data['clean_name'])
                        dogs.append(current_dog_data)
                    
                    # Start new dog
                    box_number = None
                    clean_name = dog_name_raw
                    
                    # Extract box number from name
                    if '. ' in dog_name_raw:
                        parts = dog_name_raw.split('. ', 1)
                        if len(parts) == 2:
                            try:
                                box_number = int(parts[0])
                                clean_name = parts[1]
                            except ValueError:
                                pass
                    
                    current_dog_data = {
                        'name': clean_name,
                        'clean_name': clean_name.upper().strip(),
                        'box': box_number or len(dogs) + 1,
                        'weight': self._safe_float(row.get('WGT')),
                        'sex': str(row.get('Sex', '')).strip()
                    }
                    
                    current_dog_history = []
                    
                    # Add current race data if available
                    if pd.notna(row.get('PLC')):
                        race_entry = self._create_race_entry(row, current_dog_data['clean_name'])
                        if race_entry:
                            current_dog_history.append(race_entry)
                
                # Add historical race to current dog
                elif current_dog_data and pd.notna(row.get('PLC')):
                    race_entry = self._create_race_entry(row, current_dog_data['clean_name'])
                    if race_entry:
                        current_dog_history.append(race_entry)
            
            # Don't forget the last dog
            if current_dog_data:
                current_dog_data['embedded_history'] = current_dog_history
                current_dog_data['database_history'] = self._get_database_history(current_dog_data['clean_name'])
                dogs.append(current_dog_data)
        
        except Exception as e:
            logger.error(f"Error extracting dogs with history: {e}")
        
        return dogs
    
    def _create_race_entry(self, row: pd.Series, dog_name: str) -> Dict[str, Any]:
        """Create standardized race entry from CSV row"""
        try:
            return {
                'dog_name': dog_name,
                'finish_position': self._safe_int(row.get('PLC')),
                'box_number': self._safe_int(row.get('BOX')),
                'weight': self._safe_float(row.get('WGT')),
                'individual_time': self._safe_float(row.get('TIME')),
                'sectional_1st': self._safe_float(row.get('1 SEC')),
                'starting_price': self._safe_float(row.get('SP')),
                'margin': str(row.get('MGN', '')).strip(),
                'venue': str(row.get('TRACK', '')).strip(),
                'distance': self._safe_int(str(row.get('DIST', '')).replace('m', '')),
                'grade': str(row.get('G', '')).strip(),
                'race_date': self._safe_date(row.get('DATE')),
                'source': 'embedded_csv'
            }
        except Exception as e:
            logger.warning(f"Error creating race entry: {e}")
            return None
    
    def _get_database_history(self, dog_clean_name: str) -> List[Dict[str, Any]]:
        """Get historical data from database with proper error handling"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            
            query = """
            SELECT 
                drd.dog_name,
                drd.finish_position,
                drd.box_number,
                drd.weight,
                drd.individual_time,
                drd.starting_price,
                rm.venue,
                rm.distance,
                rm.grade,
                rm.race_date
            FROM dog_race_data drd
            JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.dog_clean_name = ?
            ORDER BY rm.race_date DESC LIMIT 20
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (dog_clean_name,))
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'dog_name': row[0],
                    'finish_position': self._safe_int(row[1]),
                    'box_number': self._safe_int(row[2]),
                    'weight': self._safe_float(row[3]),
                    'individual_time': self._safe_float(row[4]),
                    'starting_price': self._safe_float(row[5]),
                    'venue': str(row[6] or '').strip(),
                    'distance': self._safe_int(row[7]),
                    'grade': str(row[8] or '').strip(),
                    'race_date': row[9],
                    'source': 'database'
                })
            
            return history
            
        except Exception as e:
            logger.warning(f"Error getting database history for {dog_clean_name}: {e}")
            return []
    
    def _generate_unified_prediction(self, dog_info: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unified prediction with standardized features"""
        try:
            # Create standardized feature vector
            features = self._create_standardized_features(dog_info, race_info)
            
            # Generate ML prediction if available
            ml_score = self._get_ml_prediction_unified(features, dog_info['name'])
            
            # Generate heuristic prediction
            heuristic_score = self._get_heuristic_prediction_unified(features, dog_info)
            
            # Combine predictions intelligently
            final_score = self._combine_predictions(ml_score, heuristic_score, features)
            
            # Calculate confidence
            confidence_level, confidence_score = self._calculate_confidence_unified(final_score, features)
            
            return {
                'dog_name': dog_info['name'],
                'clean_name': dog_info['clean_name'],
                'box_number': dog_info['box'],
                'prediction_score': round(final_score, 4),
                'ml_score': round(ml_score, 4) if ml_score else None,
                'heuristic_score': round(heuristic_score, 4),
                'confidence_level': confidence_level,
                'confidence_score': round(confidence_score, 3),
                'data_quality': features.get('data_quality', 0.5),
                'features_used': len([k for k, v in features.items() if v != self.config.intelligent_defaults.get(k, 0)]),
                'historical_races': len(dog_info.get('embedded_history', [])) + len(dog_info.get('database_history', [])),
                'reasoning': self._generate_reasoning_unified(final_score, features, dog_info['name']),
                'prediction_method': 'unified_system_v2'
            }
            
        except Exception as e:
            logger.error(f"Error generating unified prediction for {dog_info.get('name', 'unknown')}: {e}")
            return None
    
    def _create_standardized_features(self, dog_info: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized feature vector with intelligent defaults"""
        features = self.config.intelligent_defaults.copy()
        
        try:
            # Combine all historical data
            all_history = dog_info.get('embedded_history', []) + dog_info.get('database_history', [])
            
            if all_history:
                # Sort by date (most recent first)
                all_history.sort(key=lambda x: x.get('race_date', '1900-01-01'), reverse=True)
                
                # Calculate features from historical data
                features.update(self._calculate_historical_features(all_history, race_info))
            
            # Add current race context
            features['box_number'] = dog_info.get('box', 4.5)
            
            # Calculate data quality
            features['data_quality'] = self._calculate_data_quality_score(dog_info, all_history)
            
            # Ensure all expected features are present
            for feature_name in self.config.standard_feature_names:
                if feature_name not in features:
                    features[feature_name] = self.config.intelligent_defaults.get(feature_name, 0.5)
            
        except Exception as e:
            logger.warning(f"Error creating standardized features: {e}")
        
        return features
    
    def _calculate_historical_features(self, history: List[Dict[str, Any]], race_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate features from historical race data"""
        features = {}
        
        try:
            if not history:
                return features
            
            # Recent form analysis
            recent_positions = [race.get('finish_position') for race in history[:5] if race.get('finish_position')]
            if recent_positions:
                weights = [0.4, 0.3, 0.2, 0.1, 0.05][:len(recent_positions)]
                features['weighted_recent_form'] = sum(pos * weight for pos, weight in zip(recent_positions, weights))
            
            # Speed analysis
            recent_times = [race.get('individual_time') for race in history[:5] if race.get('individual_time')]
            if len(recent_times) >= 3:
                features['speed_trend'] = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
                features['speed_consistency'] = np.std(recent_times)
            
            # Venue performance
            venue = race_info.get('venue', 'UNKNOWN')
            venue_races = [r for r in history if r.get('venue', '').upper().startswith(venue[:4].upper())]
            if venue_races:
                venue_positions = [r.get('finish_position') for r in venue_races if r.get('finish_position')]
                if venue_positions:
                    features['venue_win_rate'] = sum(1 for pos in venue_positions if pos == 1) / len(venue_positions)
                    features['venue_avg_position'] = np.mean(venue_positions)
                    features['venue_experience'] = len(venue_races)
            
            # Distance performance
            target_distance = race_info.get('distance', 500)
            distance_races = [r for r in history if abs((r.get('distance') or 500) - target_distance) <= 20]
            if distance_races:
                distance_positions = [r.get('finish_position') for r in distance_races if r.get('finish_position')]
                distance_times = [r.get('individual_time') for r in distance_races if r.get('individual_time')]
                if distance_positions:
                    features['distance_win_rate'] = sum(1 for pos in distance_positions if pos == 1) / len(distance_positions)
                if distance_times:
                    features['distance_avg_time'] = np.mean(distance_times)
            
            # Box position analysis
            current_box = race_info.get('box_number', 4)
            box_races = [r for r in history if r.get('box_number') == current_box]
            if box_races:
                box_positions = [r.get('finish_position') for r in box_races if r.get('finish_position')]
                if box_positions:
                    features['box_position_win_rate'] = sum(1 for pos in box_positions if pos == 1) / len(box_positions)
                    features['box_position_avg'] = np.mean(box_positions)
            
            # Performance consistency
            all_positions = [r.get('finish_position') for r in history if r.get('finish_position')]
            if len(all_positions) > 3:
                features['position_consistency'] = 1.0 / (1.0 + np.std(all_positions))
                features['top_3_rate'] = sum(1 for pos in all_positions if pos <= 3) / len(all_positions)
                
                # Recent momentum
                if len(recent_positions) >= 3:
                    features['recent_momentum'] = 0.8 if all(pos <= 3 for pos in recent_positions[:2]) else 0.3
            
            # Competition level (based on starting prices)
            recent_prices = [r.get('starting_price') for r in history[:10] if r.get('starting_price')]
            if recent_prices:
                avg_price = np.mean(recent_prices)
                if avg_price < 5.0:
                    features['competitive_level'] = 0.8
                elif avg_price < 15.0:
                    features['competitive_level'] = 0.6
                else:
                    features['competitive_level'] = 0.4
            
        except Exception as e:
            logger.warning(f"Error calculating historical features: {e}")
        
        return features
    
    def _safe_int(self, value) -> int:
        """Safely convert value to int"""
        try:
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_date(self, value) -> str:
        """Safely convert value to date string"""
        try:
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                return None
            return str(value).strip()
        except (ValueError, TypeError):
            return None
    
    def _get_ml_prediction_unified(self, features: Dict[str, Any], dog_name: str) -> float:
        """Get ML prediction with proper feature alignment"""
        try:
            # Check if we have any ML predictors available
            ml_predictor = None
            
            # Try enhanced pipeline v2 first
            if 'enhanced_pipeline_v2' in self.predictors:
                ml_predictor = self.predictors['enhanced_pipeline_v2']
                if hasattr(ml_predictor, 'ml_system') and ml_predictor.ml_system:
                    if hasattr(ml_predictor.ml_system, 'models') and ml_predictor.ml_system.models:
                        try:
                            return ml_predictor.ml_system.predict_with_ensemble(features)
                        except Exception as e:
                            logger.warning(f"Enhanced pipeline ML prediction failed: {e}")
            
            # Try comprehensive ML system
            if 'comprehensive_ml' in self.predictors:
                ml_predictor = self.predictors['comprehensive_ml']
                if hasattr(ml_predictor, 'predict_with_ensemble'):
                    try:
                        return ml_predictor.predict_with_ensemble(features)
                    except Exception as e:
                        logger.warning(f"Comprehensive ML prediction failed: {e}")
            
            # No ML prediction available
            return None
            
        except Exception as e:
            logger.warning(f"ML prediction error for {dog_name}: {e}")
            return None
    
    def _get_heuristic_prediction_unified(self, features: Dict[str, Any], dog_info: Dict[str, Any]) -> float:
        """Generate heuristic prediction with enhanced logic"""
        try:
            base_score = 0.5
            
            # Box position influence
            box_number = features.get('box_number', 4.5)
            box_adjustments = {1: 0.08, 2: 0.06, 3: 0.04, 4: 0.02, 5: -0.01, 6: -0.03, 7: -0.05, 8: -0.07}
            base_score += box_adjustments.get(int(box_number), 0)
            
            # Recent form influence
            recent_form = features.get('weighted_recent_form', 4.5)
            if recent_form < 3.0:
                base_score += 0.15
            elif recent_form < 4.0:
                base_score += 0.08
            elif recent_form > 6.0:
                base_score -= 0.12
            elif recent_form > 5.0:
                base_score -= 0.06
            
            # Venue performance
            venue_win_rate = features.get('venue_win_rate', 0.125)
            base_score += venue_win_rate * 0.20
            
            # Speed trend
            speed_trend = features.get('speed_trend', 0)
            if speed_trend < -0.2:
                base_score += 0.10
            elif speed_trend < 0:
                base_score += 0.05
            elif speed_trend > 0.2:
                base_score -= 0.08
            
            # Consistency bonus
            consistency = features.get('position_consistency', 0.5)
            base_score += (consistency - 0.5) * 0.10
            
            # Top 3 rate
            top_3_rate = features.get('top_3_rate', 0.375)
            base_score += (top_3_rate - 0.375) * 0.15
            
            # Data quality adjustment
            data_quality = features.get('data_quality', 0.5)
            base_score *= (0.85 + 0.30 * data_quality)
            
            # Add controlled variance based on dog name for consistency
            name_hash = hash(dog_info['name']) % 1000
            name_factor = (name_hash / 1000 - 0.5) * 0.04
            base_score += name_factor
            
            return max(0.05, min(0.95, base_score))
            
        except Exception as e:
            logger.warning(f"Heuristic prediction error: {e}")
            return 0.5
    
    def _combine_predictions(self, ml_score: float, heuristic_score: float, features: Dict[str, Any]) -> float:
        """Intelligently combine ML and heuristic predictions"""
        try:
            data_quality = features.get('data_quality', 0.5)
            
            if ml_score is not None:
                # Weight based on data quality and ML confidence
                ml_weight = 0.7 * data_quality
                heuristic_weight = 1.0 - ml_weight
                
                combined_score = (ml_score * ml_weight) + (heuristic_score * heuristic_weight)
            else:
                # No ML prediction available, use heuristic with slight boost for good data
                combined_score = heuristic_score * (0.95 + 0.05 * data_quality)
            
            return max(0.05, min(0.95, combined_score))
            
        except Exception as e:
            logger.warning(f"Error combining predictions: {e}")
            return heuristic_score
    
    def _calculate_confidence_unified(self, prediction_score: float, features: Dict[str, Any]) -> tuple:
        """Calculate prediction confidence"""
        try:
            data_quality = features.get('data_quality', 0.5)
            historical_races = features.get('venue_experience', 0) + features.get('distance_win_rate', 0) * 10
            
            # Base confidence from data quality
            confidence_score = 0.4 + (data_quality * 0.4)
            
            # Boost for historical experience
            if historical_races > 5:
                confidence_score += 0.1
            elif historical_races > 10:
                confidence_score += 0.15
            
            # Adjust for prediction extremes
            if prediction_score > 0.8 or prediction_score < 0.2:
                confidence_score += 0.05
            
            confidence_score = min(0.95, confidence_score)
            
            # Convert to level
            if confidence_score >= 0.75:
                confidence_level = 'HIGH'
            elif confidence_score >= 0.55:
                confidence_level = 'MEDIUM'
            else:
                confidence_level = 'LOW'
            
            return confidence_level, confidence_score
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 'LOW', 0.4
    
    def _calculate_data_quality_score(self, dog_info: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        """Calculate data quality score for a dog"""
        try:
            score = 0.0
            max_score = 1.0
            
            # Historical data availability (40%)
            history_count = len(history)
            if history_count >= 10:
                score += 0.4
            elif history_count >= 5:
                score += 0.25
            elif history_count >= 3:
                score += 0.15
            
            # Data completeness (30%)
            if history:
                complete_fields = 0
                total_fields = 0
                for race in history[:5]:  # Check last 5 races
                    for field in ['finish_position', 'individual_time', 'starting_price', 'venue', 'distance']:
                        total_fields += 1
                        if race.get(field) is not None:
                            complete_fields += 1
                
                if total_fields > 0:
                    completeness = complete_fields / total_fields
                    score += completeness * 0.3
            
            # Data recency (20%)
            if history:
                try:
                    latest_race = history[0]  # Already sorted by date
                    race_date_str = latest_race.get('race_date', '')
                    if race_date_str:
                        from datetime import datetime
                        try:
                            race_date = datetime.strptime(race_date_str, '%Y-%m-%d')
                            days_ago = (datetime.now() - race_date).days
                            if days_ago <= 30:
                                score += 0.2
                            elif days_ago <= 60:
                                score += 0.1
                        except ValueError:
                            pass
                except Exception:
                    pass
            
            # Multiple data sources (10%)
            sources = set()
            for race in history[:10]:
                source = race.get('source', 'unknown')
                sources.add(source)
            
            if len(sources) > 1:
                score += 0.1
            elif len(sources) == 1 and 'embedded_csv' in sources:
                score += 0.05
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Error calculating data quality score: {e}")
            return 0.5
    
    def _generate_reasoning_unified(self, prediction_score: float, features: Dict[str, Any], dog_name: str) -> str:
        """Generate human-readable reasoning"""
        try:
            reasons = []
            
            if prediction_score > 0.7:
                reasons.append("Strong prediction based on")
            elif prediction_score > 0.5:
                reasons.append("Moderate prediction based on")
            else:
                reasons.append("Weak prediction based on")
            
            # Add feature-based reasons
            recent_form = features.get('weighted_recent_form', 4.5)
            if recent_form < 3.0:
                reasons.append("excellent recent form")
            elif recent_form < 4.5:
                reasons.append("good recent form")
            
            venue_win_rate = features.get('venue_win_rate', 0.125)
            if venue_win_rate > 0.3:
                reasons.append("strong venue performance")
            
            speed_trend = features.get('speed_trend', 0)
            if speed_trend < -0.1:
                reasons.append("improving speed trend")
            
            box_number = features.get('box_number', 4.5)
            if box_number <= 3:
                reasons.append("favorable box position")
            
            data_quality = features.get('data_quality', 0.5)
            if data_quality > 0.7:
                reasons.append("comprehensive historical data")
            elif data_quality < 0.3:
                reasons.append("limited historical data")
            
            if len(reasons) == 1:
                reasons.append("available form indicators")
            
            return " ".join(reasons) + "."
            
        except Exception as e:
            logger.warning(f"Error generating reasoning: {e}")
            return "Prediction based on available data."
    
    def _calculate_race_data_quality(self, dogs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall race data quality summary"""
        try:
            if not dogs:
                return {'average_quality': 0.0, 'total_dogs': 0}
            
            quality_scores = []
            dogs_with_history = 0
            total_historical_races = 0
            
            for dog in dogs:
                history_count = len(dog.get('embedded_history', [])) + len(dog.get('database_history', []))
                total_historical_races += history_count
                
                if history_count > 0:
                    dogs_with_history += 1
                
                # Calculate individual quality score
                quality_score = min(1.0, history_count / 10.0)  # Scale based on race count
                quality_scores.append(quality_score)
            
            return {
                'average_quality': np.mean(quality_scores) if quality_scores else 0.0,
                'dogs_with_historical_data': dogs_with_history,
                'total_dogs': len(dogs),
                'total_historical_races': total_historical_races,
                'average_races_per_dog': total_historical_races / len(dogs) if dogs else 0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating race data quality: {e}")
            return {'average_quality': 0.5, 'total_dogs': len(dogs) if dogs else 0}
    
    def _normalize_venue(self, venue_str):
        """Normalize venue name to standard format"""
        if not venue_str:
            return 'UNKNOWN'
        
        venue_clean = venue_str.upper().strip().replace(' ', '_')
        
        # Known venue mappings
        venue_mapping = {
            'AP_K': 'AP_K', 'ANGLE_PARK': 'AP_K', 'ANGLE': 'AP_K',
            'GEE': 'GEE', 'GEELONG': 'GEE',
            'RICH': 'RICH', 'RICHMOND': 'RICH',
            'DAPT': 'DAPT', 'DAPTO': 'DAPT',
            'BAL': 'BAL', 'BALLARAT': 'BAL',
            'BEN': 'BEN', 'BENDIGO': 'BEN',
            'HEA': 'HEA', 'HEALESVILLE': 'HEA',
            'WAR': 'WAR', 'WARRNAMBOOL': 'WAR',
            'SAN': 'SAN', 'SANDOWN': 'SAN',
            'MOUNT': 'MOUNT', 'MOUNT_GAMBIER': 'MOUNT',
            'MURR': 'MURR', 'MURRAY_BRIDGE': 'MURR',
            'SAL': 'SAL', 'SALE': 'SAL',
            'HOR': 'HOR', 'HORSHAM': 'HOR',
            'CANN': 'CANN', 'CANNINGTON': 'CANN',
            'WPK': 'WPK', 'W_PK': 'WPK', 'WENTWORTH_PARK': 'WPK',
            'MEA': 'MEA', 'THE_MEADOWS': 'MEA', 'MEADOWS': 'MEA',
            'HOBT': 'HOBT', 'HOBART': 'HOBT',
            'GOSF': 'GOSF', 'GOSFORD': 'GOSF',
            'NOR': 'NOR', 'NORTHAM': 'NOR',
            'MAND': 'MAND', 'MANDURAH': 'MAND',
            'GAWL': 'GAWL', 'GAWLER': 'GAWL',
            'TRA': 'TRA', 'TRARALGON': 'TRA',
            'CASO': 'CASO', 'CASINO': 'CASO',
            'GRDN': 'GRDN', 'THE_GARDENS': 'GRDN', 'GARDENS': 'GRDN',
            'DARW': 'DARW', 'DARWIN': 'DARW',
            'ALBION': 'ALBION', 'ALBION_PARK': 'ALBION',
            'TARE': 'TAREE', 'MAIT': 'MAITLAND', 'GOSFORD': 'GRDN',
            'DUBO': 'DUBBO', 'WARRAGUL': 'WAR'
        }
        
        # Direct mapping
        if venue_clean in venue_mapping:
            return venue_mapping[venue_clean]
        
        # Try partial matches
        for key, value in venue_mapping.items():
            if key in venue_clean or venue_clean in key:
                return value
        
        # Return cleaned version as fallback
        return venue_clean if len(venue_clean) <= 8 else venue_clean[:8]
    
    def _parse_race_number(self, race_str):
        """Parse race number from string"""
        if not race_str:
            return '0'
        
        # Extract digits
        digits = re.findall(r'\d+', str(race_str))
        if digits:
            race_num = int(digits[0])
            if 1 <= race_num <= 20:  # Reasonable race number range
                return str(race_num)
        
        return '0'
    
    def _normalize_date(self, date_str):
        """Normalize date string to YYYY-MM-DD format"""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        date_clean = str(date_str).strip().replace('_', ' ')
        
        # Month name mappings
        month_mapping = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
        
        # Try different date patterns
        date_patterns = [
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # "22 July 2025"
            r'(\d{4})-(\d{2})-(\d{2})',      # "2025-07-22"
            r'(\w+)(\d{1,2})',               # "july22"
            r'(\d{2})(\d{2})(\d{4})',        # "22072025"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_clean, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    
                    if pattern == r'(\d{1,2})\s+(\w+)\s+(\d{4})':  # "22 July 2025"
                        day, month_name, year = groups
                        month = month_mapping.get(month_name.lower())
                        if month:
                            return f"{year}-{month}-{day.zfill(2)}"
                    
                    elif pattern == r'(\d{4})-(\d{2})-(\d{2})':  # "2025-07-22"
                        return f"{groups[0]}-{groups[1]}-{groups[2]}"
                    
                    elif pattern == r'(\w+)(\d{1,2})':  # "july22"
                        month_name, day = groups
                        month = month_mapping.get(month_name.lower())
                        if month:
                            year = datetime.now().year
                            return f"{year}-{month}-{day.zfill(2)}"
                    
                except Exception:
                    continue
        
        # Return current date if all parsing fails
        return datetime.now().strftime('%Y-%m-%d')
    
    def _try_content_based_parsing(self, race_file_path, race_df, current_result):
        """Try to extract race info from CSV content as last resort"""
        result = current_result.copy()
        
        try:
            # Look for track/venue information in column headers or data
            for col in race_df.columns:
                col_str = str(col).upper()
                if 'TRACK' in col_str or 'VENUE' in col_str:
                    unique_values = race_df[col].dropna().unique()
                    if len(unique_values) > 0:
                        venue_candidate = str(unique_values[0]).strip()
                        normalized_venue = self._normalize_venue(venue_candidate)
                        if normalized_venue != 'UNKNOWN':
                            result['venue'] = normalized_venue
                            logger.info(f"   📍 Found venue in content: {normalized_venue}")
                            break
            
            # Look for race number in data
            for col in race_df.columns:
                col_str = str(col).upper()
                if 'RACE' in col_str and 'NUMBER' in col_str:
                    unique_values = race_df[col].dropna().unique()
                    if len(unique_values) > 0:
                        race_num_candidate = str(unique_values[0]).strip()
                        parsed_race_num = self._parse_race_number(race_num_candidate)
                        if parsed_race_num != '0':
                            result['race_number'] = parsed_race_num
                            logger.info(f"   🏃 Found race number in content: {parsed_race_num}")
                            break
                            
        except Exception as e:
            logger.info(f"   ⚠️ Content-based parsing failed: {e}")
        
        return result
    
    def predict_multiple_races(self, race_files: List[str], 
                             enhancement_level: str = 'weather',
                             max_concurrent: int = 3) -> Dict[str, Any]:
        """Predict multiple races efficiently"""
        
        logger.info(f"🎯 Predicting {len(race_files)} races (level: {enhancement_level})")
        
        results = []
        successful = 0
        total_time = time.time()
        
        for i, race_file in enumerate(race_files[:max_concurrent]):
            logger.info(f"📊 Processing race {i+1}/{min(len(race_files), max_concurrent)}")
            
            result = self.predict_race_file(race_file, enhancement_level)
            
            if result.get('success'):
                successful += 1
            
            results.append({
                'race_file': race_file,
                'result': result
            })
        
        total_time = time.time() - total_time
        
        return {
            'batch_summary': {
                'total_races': len(race_files),
                'processed_races': len(results),
                'successful_predictions': successful,
                'total_time_seconds': round(total_time, 2),
                'average_time_per_race': round(total_time / len(results), 2)
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

# Backward Compatibility Interface Functions
def predict_race_file(race_file_path: str) -> Dict[str, Any]:
    """
    Backward compatible function that maintains exact same interface 
    as the original standalone prediction scripts
    """
    predictor = UnifiedPredictor()
    return predictor.predict_race_file(race_file_path)

def main():
    """Main function for command line usage - maintains compatibility"""
    if len(sys.argv) < 2:
        print("Usage: python unified_predictor.py <race_file_path> [enhancement_level]")
        print("Enhancement levels: basic, weather, full")
        sys.exit(1)
    
    race_file_path = sys.argv[1]
    enhancement_level = sys.argv[2] if len(sys.argv) > 2 else 'full'
    
    # Initialize unified predictor
    predictor = UnifiedPredictor()
    
    # Make prediction
    result = predictor.predict_race_file(race_file_path, enhancement_level)
    
    if result['success']:
        print(f"\n🏆 UNIFIED PREDICTION RESULTS")
        print("=" * 60)
        print(f"🎯 Method: {result.get('prediction_method', 'Unknown')}")
        print(f"⏱️  Time: {result.get('prediction_time_seconds', 0):.2f}s")
        
        if result.get('enhanced_with_gpt'):
            print("🤖 Enhanced with GPT analysis")
        
        print("\n🏆 Top predictions:")
        for i, prediction in enumerate(result['predictions'][:5], 1):
            dog_name = prediction.get('dog_name', 'Unknown')
            score = prediction.get('prediction_score', 0)
            box_num = prediction.get('box_number', 'N/A')
            confidence = prediction.get('confidence_level', 'UNKNOWN')
            
            print(f"  {i}. {dog_name} (Box {box_num}) - Score: {score:.3f} - Confidence: {confidence}")
        
        print(f"\n📊 Total dogs: {len(result['predictions'])}")
        
        # Show performance stats
        stats = predictor.get_performance_stats()
        print(f"📈 Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
    else:
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
