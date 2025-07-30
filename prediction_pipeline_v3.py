"""
Prediction Pipeline V3 - Comprehensive Integrated System
========================================================

A comprehensive prediction pipeline that integrates ALL available analysis methods:
- ML System V3 (primary)
- Weather-Enhanced Predictions (when available)
- GPT Analysis Integration (when available)
- Comprehensive Feature Engineering
- Unified Predictor (as enhanced fallback)
- Traditional Analysis (as basic fallback)

This is the primary prediction interface with intelligent fallback hierarchy.
"""

import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from ml_system_v3 import MLSystemV3

# Import comprehensive analysis systems (with fallback handling)
# Note: These modules have been moved to archive/outdated_scripts/ but are maintained for backward compatibility
import sys
sys.path.append('archive/outdated_scripts')

try:
    from archive.outdated_scripts.weather_enhanced_predictor import WeatherEnhancedPredictor
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False
    
try:
    from archive.outdated_scripts.gpt_prediction_enhancer import GPTPredictionEnhancer
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    
try:
    from archive.outdated_scripts.comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
    COMPREHENSIVE_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_AVAILABLE = False

# Note: unified_predictor is kept in root as it's still actively used as fallback
try:
    from unified_predictor import UnifiedPredictor
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False

logger = logging.getLogger(__name__)

class PredictionPipelineV3:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.ml_system = MLSystemV3(db_path)
        
        # Initialize advanced systems
        self.weather_predictor = None
        self.gpt_enhancer = None
        self.unified_predictor = None
        self.comprehensive_pipeline = None
        
        self._initialize_advanced_systems()
        
        # Print system status
        print("🚀 Prediction Pipeline V3 - Comprehensive System Initialized")
        self._print_system_status()
    
    def _initialize_advanced_systems(self):
        """Initialize all available advanced prediction systems"""
        if WEATHER_AVAILABLE:
            try:
                self.weather_predictor = WeatherEnhancedPredictor(self.db_path)
                logger.info("✅ Weather Enhanced Predictor initialized")
            except Exception as e:
                logger.warning(f"⚠️ Weather predictor initialization failed: {e}")
        
        if GPT_AVAILABLE:
            try:
                self.gpt_enhancer = GPTPredictionEnhancer(self.db_path)
                logger.info("✅ GPT Prediction Enhancer initialized")
            except Exception as e:
                logger.warning(f"⚠️ GPT enhancer initialization failed: {e}")
        
        if UNIFIED_AVAILABLE:
            try:
                self.unified_predictor = UnifiedPredictor()
                logger.info("✅ Unified Predictor initialized")
            except Exception as e:
                logger.warning(f"⚠️ Unified predictor initialization failed: {e}")
        
        if COMPREHENSIVE_AVAILABLE:
            try:
                self.comprehensive_pipeline = ComprehensivePredictionPipeline(self.db_path)
                logger.info("✅ Comprehensive Pipeline initialized")
            except Exception as e:
                logger.warning(f"⚠️ Comprehensive pipeline initialization failed: {e}")
    
    def _print_system_status(self):
        """Print status of all available systems"""
        print("\n🔧 Integrated System Status:")
        print(f"  ML System V3: ✅ (Primary)")
        print(f"  Weather Enhanced: {'✅' if self.weather_predictor else '❌'}")
        print(f"  GPT Enhancement: {'✅' if self.gpt_enhancer else '❌'}")
        print(f"  Unified Predictor: {'✅' if self.unified_predictor else '❌'}")
        print(f"  Comprehensive Pipeline: {'✅' if self.comprehensive_pipeline else '❌'}")
        
        available_systems = 1 + sum([
            self.weather_predictor is not None,
            self.gpt_enhancer is not None,
            self.unified_predictor is not None,
            self.comprehensive_pipeline is not None
        ])
        
        print(f"  Overall: {available_systems}/5 systems available")
    
    def predict_race_file(self, race_file_path: str, enhancement_level='full') -> dict:
        """Main prediction method with intelligent fallback and enhancement levels."""
        
        logger.info(f"🚀 Starting comprehensive prediction for: {os.path.basename(race_file_path)}")
        logger.info(f"   Enhancement level: {enhancement_level}")
        
        # --- Primary Method: Comprehensive Pipeline ---
        if self.comprehensive_pipeline and enhancement_level == 'full':
            try:
                logger.info("  -> Trying Comprehensive Prediction Pipeline (V3 Primary)...")
                result = self.comprehensive_pipeline.predict_race_file_with_all_enhancements(race_file_path)
                if result and result.get('success'):
                    logger.info("  ✅ Comprehensive prediction successful!")
                    return result
                else:
                    logger.warning("  ⚠️ Comprehensive prediction failed. Falling back...")
            except Exception as e:
                logger.error(f"  ❌ Comprehensive pipeline error: {e}. Falling back...")

        # --- Fallback 1: Weather-Enhanced Predictor ---
        if self.weather_predictor:
            try:
                logger.info("  -> Trying Weather-Enhanced Predictor...")
                result = self.weather_predictor.predict_race_file_with_weather(race_file_path)
                if result and result.get('success'):
                    logger.info("  ✅ Weather-enhanced prediction successful!")
                    return result
                else:
                    logger.warning("  ⚠️ Weather-enhanced prediction failed. Falling back...")
            except Exception as e:
                logger.error(f"  ❌ Weather-enhanced predictor error: {e}. Falling back...")

        # --- Fallback 2: Unified Predictor ---
        if self.unified_predictor:
            try:
                logger.info("  -> Trying Unified Predictor...")
                result = self.unified_predictor.predict_race_file(race_file_path)
                if result and result.get('success'):
                    logger.info("  ✅ Unified predictor successful!")
                    return result
                else:
                    logger.warning("  ⚠️ Unified predictor failed. Falling back...")
            except Exception as e:
                logger.error(f"  ❌ Unified predictor error: {e}. Falling back...")

        # --- Fallback 3: Basic ML System V3 (as a last resort) ---
        logger.info("  -> Trying Basic ML System V3 (Final Fallback)...")
        try:
            # This is the original simple prediction logic
            race_df = self._load_race_file(race_file_path)
            if race_df is None or race_df.empty:
                return self._error_response("Could not load race file or file is empty")
            
            dogs = self._extract_dogs(race_df, race_file_path)
            if not dogs:
                return self._error_response("No participating dogs found in race file")
            
            predictions = []
            for dog in dogs:
                ml_result = self.ml_system.predict(dog)
                prediction = {
                    'dog_name': dog['name'],
                    'box_number': dog.get('box_number', 0),
                    'win_probability': ml_result['win_probability'],
                    'prediction_method': 'ml_system_v3_basic'
                }
                predictions.append(prediction)
            
            if not predictions:
                return self._error_response("No predictions could be generated by basic ML system")
            
            predictions.sort(key=lambda x: x['win_probability'], reverse=True)
            
            return {
                'success': True,
                'predictions': predictions,
                'prediction_method': 'ml_system_v3_basic',
                'note': 'Used basic ML system as a final fallback'
            }

        except Exception as e:
            logger.critical(f"  ❌ All prediction methods failed. Final error: {e}")
            return self._error_response(f"All prediction methods failed. Final error: {e}")
    
    def _load_race_file(self, race_file_path: str) -> pd.DataFrame:
        """Load and parse race file"""
        try:
            # Better delimiter detection
            with open(race_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ',' if first_line.count(',') > first_line.count('|') else '|'
            
            df = pd.read_csv(race_file_path, delimiter=delimiter)
            logger.debug(f"Loaded race file with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading race file: {e}")
            return None
    
    def _extract_dogs(self, race_df: pd.DataFrame, race_file_path: str) -> list:
        """Extract dog information from race file"""
        dogs = []
        
        try:
            current_dog = None
            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get('Dog Name', '')).strip()
                
                # Check if this is a new dog entry
                if dog_name_raw and dog_name_raw not in ['""', '', 'nan', 'NaN']:
                    # Save previous dog
                    if current_dog:
                        dogs.append(current_dog)
                    
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
                    
                    current_dog = {
                        'name': clean_name,
                        'box_number': box_number or len(dogs) + 1,
                        'weight': float(row.get('WGT', 30.0)) if pd.notna(row.get('WGT')) else 30.0,
                        'starting_price': float(row.get('SP', 10.0)) if pd.notna(row.get('SP')) else 10.0,
                        'individual_time': float(row.get('TIME', 30.0)) if pd.notna(row.get('TIME')) else 30.0,
                        'field_size': 8,  # Default field size
                        'temperature': 20.0,  # Default values
                        'humidity': 60.0,
                        'wind_speed': 10.0
                    }
            
            # Don't forget the last dog
            if current_dog:
                dogs.append(current_dog)
                
        except Exception as e:
            logger.error(f"Error extracting dogs: {e}")
        
        return dogs
    
    def _generate_reasoning(self, ml_result: dict, dog: dict) -> str:
        """Generate human-readable reasoning"""
        win_prob = ml_result.get('final_score', ml_result.get('win_probability', 0.0))
        confidence = ml_result.get('confidence', 0.5)
        
        if win_prob > 0.7:
            strength = "Strong"
        elif win_prob > 0.5:
            strength = "Moderate"
        elif win_prob > 0.3:
            strength = "Fair"
        else:
            strength = "Weak"
        
        conf_text = "high confidence" if confidence > 0.7 else "medium confidence" if confidence > 0.4 else "low confidence"
        
        factors = []
        if dog.get('starting_price', 10) < 3.0:
            factors.append("market favorite")
        if dog.get('box_number', 4) <= 3:
            factors.append("inside box")
        if dog.get('weight', 30) > 32:
            factors.append("heavy weight")
        
        factor_text = f" based on {', '.join(factors)}" if factors else ""
        
        return f"{strength} prediction with {conf_text}{factor_text}."
    
    def _validate_predictions(self, predictions: list) -> list:
        """Validate prediction quality"""
        issues = []
        
        if not predictions:
            issues.append("No predictions generated")
            return issues
        
        # Check for reasonable prediction spread
        win_probs = [p.get('final_score', p.get('win_probability', 0.0)) for p in predictions]
        prob_range = max(win_probs) - min(win_probs)
        
        if prob_range < 0.1:
            issues.append(f"Win probabilities too similar (range: {prob_range:.3f})")
        
        if max(win_probs) < 0.2:
            issues.append(f"All win probabilities very low (max: {max(win_probs):.3f})")
        
        if min(win_probs) > 0.8:
            issues.append(f"All win probabilities very high (min: {min(win_probs):.3f})")
        
        return issues
    
    def _extract_venue_from_filename(self, filename: str) -> str:
        """Extract venue from filename"""
        try:
            basename = os.path.basename(filename)
            if ' - ' in basename:
                parts = basename.split(' - ')
                if len(parts) >= 2:
                    return parts[1]
            return "UNKNOWN"
        except:
            return "UNKNOWN"
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename"""
        try:
            basename = os.path.basename(filename)
            import re
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, basename)
            if match:
                return match.group(1)
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')
    
    def _error_response(self, error_message: str) -> dict:
        """Generate error response"""
        return {
            'success': False,
            'error': error_message,
            'predictions': [],
            'prediction_method': 'ml_system_v3'
        }

# Function for frontend training calls
def train_new_model(model_type='gradient_boosting'):
    """Train a new ML model - can be called from frontend"""
    try:
        ml_system = MLSystemV3()
        success = ml_system.train_model(model_type)
        
        if success:
            return {
                'success': True,
                'message': 'Model trained successfully',
                'model_info': ml_system.get_model_info()
            }
        else:
            return {
                'success': False,
                'message': 'Model training failed'
            }
    except Exception as e:
        return {
            'success': False,
            'message': f'Training error: {str(e)}'
        }
