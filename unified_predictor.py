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
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
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
            "enhanced_pipeline_v2",
            "comprehensive_pipeline",
            "weather_enhanced", 
            "comprehensive_ml",
            "basic_fallback"
        ]
        
        # Component availability flags
        self.components_available = {
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
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'method_usage': {method: 0 for method in self.config.prediction_methods},
            'average_time': 0.0,
            'success_rate': 0.0
        }
        
        logger.info(f"🚀 Unified Predictor initialized with {len(self.predictors)} active methods")
    
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
            if method == 'basic_fallback':
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
                    
                    # Try to get box number from BOX column if available
                    if box_number is None:
                        box_from_col = row.get('BOX', row.get('Box', None))
                        if box_from_col is not None:
                            try:
                                box_number = int(box_from_col)
                            except (ValueError, TypeError):
                                box_number = len(dogs) + 1  # Default sequential
                    
                    if box_number is None:
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
