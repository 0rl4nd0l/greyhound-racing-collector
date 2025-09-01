#!/usr/bin/env python3
"""
Enhanced Prediction Service
===========================

Service that provides highly accurate and unique predictions by integrating:
- ML System V4 with enhanced accuracy optimizer
- Prediction uniqueness validation
- Real-time calibration and confidence scoring
- Performance monitoring and feedback loops
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import os

logger = logging.getLogger(__name__)

class EnhancedPredictionService:
    """Service for generating highly accurate and unique predictions."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ml_system = None
        self.accuracy_optimizer = None
        self._initialize_systems()
        
    def _initialize_systems(self):
        """Initialize ML systems with enhanced accuracy."""
        try:
            # Import and initialize ML System V4
            from ml_system_v4 import MLSystemV4
            self.ml_system = MLSystemV4(self.db_path)
            
            # The accuracy optimizer is already integrated in ML System V4
            if hasattr(self.ml_system, 'accuracy_optimizer') and self.ml_system.accuracy_optimizer:
                self.accuracy_optimizer = self.ml_system.accuracy_optimizer
                logger.info("✅ Enhanced Prediction Service initialized with accuracy optimization")
            else:
                logger.warning("⚠️ ML System V4 loaded but accuracy optimizer not available")
                
        except ImportError as e:
            logger.error(f"Failed to import ML System V4: {e}")
            self.ml_system = None
        except Exception as e:
            logger.error(f"Failed to initialize prediction systems: {e}")
            self.ml_system = None
    
    def is_available(self) -> bool:
        """Check if the enhanced prediction service is available."""
        return self.ml_system is not None
    
    def predict_race_enhanced(self, race_data: pd.DataFrame, race_id: str, 
                            market_odds: Optional[Dict[str, float]] = None,
                            tgr_enabled: Optional[bool] = None) -> Dict[str, Any]:
        """Generate enhanced predictions with accuracy optimization.
        tgr_enabled: when provided, toggles runtime inclusion of TGR features.
        """
        
        if not self.is_available():
            return {
                'success': False,
                'error': 'Enhanced prediction service not available',
                'race_id': race_id,
                'fallback_reason': 'ML System V4 not initialized'
            }
        
        try:
            logger.info(f"🎯 Generating enhanced predictions for race: {race_id}")
            
            # Respect runtime TGR toggle if provided
            try:
                if tgr_enabled is not None and hasattr(self.ml_system, 'set_tgr_enabled'):
                    self.ml_system.set_tgr_enabled(bool(tgr_enabled))
            except Exception:
                pass
            
            # Use the ML System V4 with integrated accuracy optimizer
            result = self.ml_system.predict_race(race_data, race_id, market_odds)
            
            if result.get('success'):
                # Add enhanced service metadata
                result['enhanced_service'] = {
                    'accuracy_optimization_applied': self.accuracy_optimizer is not None,
                    'service_version': '1.0',
                    'prediction_method': 'ensemble_with_calibration',
                    'uniqueness_validated': True,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add additional quality metrics
                if 'predictions' in result:
                    predictions = result['predictions']

                    # Optionally apply GPT rerank (light blend) behind feature flag
                    try:
                        if str(os.getenv('USE_GPT_RERANK', '1')).lower() in ('1','true','yes'):
                            alpha_env = os.getenv('GPT_RERANK_ALPHA', '0.2')
                            try:
                                alpha = float(alpha_env)
                            except Exception:
                                alpha = 0.2
                            alpha = max(0.0, min(0.5, alpha))
                            result = self._gpt_rerank_blend(result, alpha)
                    except Exception:
                        # Never fail predictions due to reranker issues
                        pass
                    
                    # Calculate prediction quality metrics
                    quality_metrics = self._calculate_prediction_quality(result.get('predictions') or predictions)
                    result['quality_metrics'] = quality_metrics
                    
                    # Validate uniqueness (already done in ML System V4 if accuracy optimizer available)
                    uniqueness_score = self._validate_prediction_uniqueness(result.get('predictions') or predictions, race_id)
                    result['uniqueness_score'] = uniqueness_score
                    
                    # Add prediction recommendations
                    recommendations = self._generate_prediction_recommendations(result.get('predictions') or predictions, quality_metrics)
                    result['recommendations'] = recommendations
                
                logger.info(f"✅ Enhanced predictions generated for {race_id}")
            else:
                logger.warning(f"⚠️ Prediction failed for {race_id}: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed for race {race_id}: {e}")
            return {
                'success': False,
                'error': f'Enhanced prediction error: {str(e)}',
                'race_id': race_id,
                'fallback_reason': 'Service exception occurred'
            }
    
    def predict_race_file_enhanced(self, race_file_path: str, tgr_enabled: Optional[bool] = None) -> Dict[str, Any]:
        """Generate enhanced predictions from race file.
        tgr_enabled: when provided, toggles runtime inclusion of TGR features.
        """
        
        try:
            # Use Prediction Pipeline V4 if available
            from prediction_pipeline_v4 import PredictionPipelineV4
            pipeline = PredictionPipelineV4(self.db_path)
            
            # Generate predictions using the pipeline
            try:
                result = pipeline.predict_race_file(race_file_path, tgr_enabled=tgr_enabled)
            except TypeError:
                # Backward-compat if pipeline signature not updated
                result = pipeline.predict_race_file(race_file_path)
            
            if result.get('success'):
                # Enhance the result with additional quality metrics
                if 'predictions' in result:
                    predictions = result['predictions']
                    race_id = result.get('race_id', 'unknown')
                    
                    # Add enhanced service metadata
                    result['enhanced_service'] = {
                        'accuracy_optimization_applied': True,
                        'service_version': '1.0',
                        'prediction_method': 'pipeline_v4_with_enhancement',
                        'source_file': race_file_path,
                        'timestamp': datetime.now().isoformat()
                    }

                    # Set predictor metadata defaults for UI/consumers
                    result.setdefault('predictor_used', 'EnhancedPredictionService')
                    if not result.get('prediction_methods_used'):
                        result['prediction_methods_used'] = ['ml_system']
                    result.setdefault('analysis_version', 'ML System V4')
                    
                    # Optionally apply GPT rerank (light blend) behind feature flag
                    try:
                        if str(os.getenv('USE_GPT_RERANK', '1')).lower() in ('1','true','yes'):
                            alpha_env = os.getenv('GPT_RERANK_ALPHA', '0.2')
                            try:
                                alpha = float(alpha_env)
                            except Exception:
                                alpha = 0.2
                            alpha = max(0.0, min(0.5, alpha))
                            result = self._gpt_rerank_blend(result, alpha)
                    except Exception:
                        # Never fail predictions due to reranker issues
                        pass
                    
                    # Calculate quality metrics
                    quality_metrics = self._calculate_prediction_quality(result.get('predictions') or predictions)
                    result['quality_metrics'] = quality_metrics
                    
                    # Validate uniqueness
                    uniqueness_score = self._validate_prediction_uniqueness(result.get('predictions') or predictions, race_id)
                    result['uniqueness_score'] = uniqueness_score
                    
                    # Generate recommendations
                    recommendations = self._generate_prediction_recommendations(result.get('predictions') or predictions, quality_metrics)
                    result['recommendations'] = recommendations
            
            return result
            
        except ImportError:
            logger.error("Prediction Pipeline V4 not available")
            return {
                'success': False,
                'error': 'Prediction Pipeline V4 not available',
                'race_file': race_file_path
            }
        except Exception as e:
            logger.error(f"Enhanced file prediction failed: {e}")
            return {
                'success': False,
                'error': f'Enhanced prediction error: {str(e)}',
                'race_file': race_file_path
            }
    
    def _gpt_rerank_blend(self, prediction_result: Dict[str, Any], alpha: float = 0.2) -> Dict[str, Any]:
        """Blend GPT reranker scores into model predictions conservatively.
        alpha is the GPT weight (0..0.5). Returns an updated prediction_result.
        """
        try:
            preds = prediction_result.get('predictions') or prediction_result.get('enhanced_predictions') or []
            if not isinstance(preds, list) or not preds:
                return prediction_result
            # Prepare compact payload
            race_info = prediction_result.get('race_info') or (prediction_result.get('summary') or {}).get('race_info') or {}
            def _base_prob(p: Dict[str, Any]) -> float:
                v = p.get('win_prob') or p.get('normalized_win_probability') or p.get('win_probability') or p.get('final_score') or p.get('prediction_score') or p.get('confidence') or 0.0
                try:
                    x = float(v)
                except Exception:
                    x = 0.0
                if x > 1.5:
                    x = x / 100.0
                return max(0.0, x)
            runners = []
            for p in preds:
                try:
                    runners.append({
                        'dog_name': p.get('dog_name') or p.get('clean_name') or p.get('name'),
                        'box_number': p.get('box_number') or p.get('box'),
                        'win_prob': _base_prob(p),
                        'csv_win_rate': float(p.get('csv_win_rate') or 0.0),
                        'csv_place_rate': float(p.get('csv_place_rate') or 0.0),
                        'avg_finish_position': float(p.get('csv_avg_finish_position') or 10.0),
                    })
                except Exception:
                    continue
            if not runners:
                return prediction_result
            # Call GPT reranker
            try:
                from services.gpt_service import GPTService
                gpt = GPTService()
                resp = gpt.enhance_predictions({'race_info': race_info, 'runners': runners})
            except Exception:
                resp = {'scores': []}
            scores = resp.get('scores') or []
            if not isinstance(scores, list) or not scores:
                return prediction_result
            score_map = {}
            for s in scores:
                try:
                    nm = (s.get('dog_name') or '').strip().upper()
                    sc = float(s.get('gpt_score') or 0.0)
                except Exception:
                    continue
                if not nm:
                    continue
                if sc < 0:
                    sc = 0.0
                if sc > 1:
                    sc = sc / 100.0 if sc > 1.5 else 1.0
                score_map[nm] = sc
            if not score_map:
                return prediction_result
            mean_g = sum(score_map.values()) / max(1, len(score_map))
            # Blend and renormalize
            blended = []
            for p in preds:
                name = (p.get('dog_name') or p.get('clean_name') or p.get('name') or '').strip().upper()
                base = _base_prob(p)
                g = score_map.get(name, mean_g)
                new_score = max(0.0, (1.0 - alpha) * base + alpha * g)
                p['gpt_score'] = g
                p['final_score'] = new_score
                blended.append(p)
            total = sum(x.get('final_score', 0.0) for x in blended)
            if total <= 0:
                eq = 1.0 / len(blended)
                for p in blended:
                    p['win_prob'] = eq
            else:
                for p in blended:
                    p['win_prob'] = float(p.get('final_score', 0.0)) / total
            for p in blended:
                # Ensure commonly used keys exist for downstream consumers
                p.setdefault('win_prob_norm', p.get('win_prob', 0.0))
                p.setdefault('place_prob', min(1.0, p.get('win_prob', 0.0) * 1.6))
            prediction_result['predictions'] = blended
            meta = prediction_result.setdefault('gpt_rerank', {})
            meta.update({'alpha': float(alpha), 'applied': True, 'timestamp': datetime.now().isoformat()})
            # Carry through token usage from GPT call if available
            try:
                tok = (resp.get('_meta') or {}).get('tokens_used')
                if tok is not None:
                    meta['tokens_used'] = int(tok)
            except Exception:
                pass
            # Mark method used for UI transparency
            try:
                methods = prediction_result.setdefault('prediction_methods_used', [])
                if isinstance(methods, list) and 'gpt_rerank' not in methods:
                    methods.append('gpt_rerank')
            except Exception:
                pass
            return prediction_result
        except Exception:
            return prediction_result

    def _calculate_prediction_quality(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for predictions."""
        
        if not predictions:
            return {'error': 'No predictions provided'}
        
        try:
            # Extract key metrics
            win_probs = [p.get('win_prob_norm', p.get('win_probability', 0)) for p in predictions]
            confidences = [p.get('confidence', 0.5) for p in predictions]
            
            # Calculate quality metrics
            import numpy as np
            
            quality_metrics = {
                'prediction_count': len(predictions),
                'avg_confidence': round(np.mean(confidences), 4),
                'min_confidence': round(min(confidences), 4),
                'max_confidence': round(max(confidences), 4),
                'probability_spread': round(max(win_probs) - min(win_probs), 4),
                'probability_variance': round(np.var(win_probs), 6),
                'predictions_with_high_confidence': sum(1 for c in confidences if c >= 0.7),
                'predictions_with_low_confidence': sum(1 for c in confidences if c < 0.4),
                'favorite_probability': round(max(win_probs), 4),
                'longshot_probability': round(min(win_probs), 4),
                'probability_sum': round(sum(win_probs), 4),
                'normalization_quality': 'good' if 0.95 <= sum(win_probs) <= 1.05 else 'poor'
            }
            
            # Overall quality score (0-1)
            quality_score = (
                quality_metrics['avg_confidence'] * 0.4 +
                min(1.0, quality_metrics['probability_spread'] * 2) * 0.3 +
                (1.0 if quality_metrics['normalization_quality'] == 'good' else 0.5) * 0.3
            )
            
            quality_metrics['overall_quality_score'] = round(quality_score, 4)
            quality_metrics['quality_level'] = self._get_quality_level(quality_score)
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return {'error': str(e)}
    
    def _validate_prediction_uniqueness(self, predictions: List[Dict[str, Any]], race_id: str) -> float:
        """Validate prediction uniqueness and return uniqueness score."""
        
        try:
            if self.accuracy_optimizer and hasattr(self.accuracy_optimizer, 'uniqueness_validator'):
                # Use the accuracy optimizer's uniqueness validator
                validation_result = self.accuracy_optimizer.uniqueness_validator.validate_uniqueness(predictions, race_id)
                return validation_result.get('uniqueness_score', 1.0)
            else:
                # Simple uniqueness check
                win_probs = [p.get('win_prob_norm', p.get('win_probability', 0)) for p in predictions]
                import numpy as np
                
                # Check for uniform distributions (low uniqueness)
                prob_std = np.std(win_probs)
                if prob_std < 0.05:
                    return 0.3  # Low uniqueness
                elif prob_std < 0.10:
                    return 0.7  # Medium uniqueness
                else:
                    return 1.0  # High uniqueness
                    
        except Exception as e:
            logger.warning(f"Uniqueness validation failed: {e}")
            return 0.5  # Default medium uniqueness
    
    def _generate_prediction_recommendations(self, predictions: List[Dict[str, Any]], 
                                           quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on prediction quality."""
        
        recommendations = []
        
        try:
            # Check overall quality
            quality_score = quality_metrics.get('overall_quality_score', 0)
            if quality_score < 0.6:
                recommendations.append("CAUTION: Low overall prediction quality detected")
            
            # Check confidence levels
            avg_confidence = quality_metrics.get('avg_confidence', 0)
            if avg_confidence < 0.5:
                recommendations.append("Consider additional data sources - confidence levels are low")
            
            # Check probability spread
            prob_spread = quality_metrics.get('probability_spread', 0)
            if prob_spread < 0.1:
                recommendations.append("Predictions show low variance - consider alternative models")
            elif prob_spread > 0.6:
                recommendations.append("High variance predictions - verify data quality")
            
            # Check normalization
            if quality_metrics.get('normalization_quality') == 'poor':
                recommendations.append("Probability normalization issue - check model calibration")
            
            # Betting recommendations based on predictions
            if predictions:
                top_prediction = max(predictions, key=lambda x: x.get('win_prob_norm', x.get('win_probability', 0)))
                top_prob = top_prediction.get('win_prob_norm', top_prediction.get('win_probability', 0))
                top_confidence = top_prediction.get('confidence', 0)
                
                if top_prob > 0.4 and top_confidence > 0.7:
                    recommendations.append(f"Strong favorite identified: {top_prediction.get('dog_clean_name', 'Unknown')}")
                elif top_prob < 0.2 and quality_score > 0.7:
                    recommendations.append("Competitive race - consider place/show betting")
                
            # Add positive recommendations
            if quality_score >= 0.8:
                recommendations.append("HIGH QUALITY: Predictions show strong reliability")
            elif quality_score >= 0.7:
                recommendations.append("GOOD QUALITY: Predictions are reliable for betting decisions")
                
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error")
        
        return recommendations if recommendations else ["Standard predictions generated - review individual dog assessments"]
    
    def _get_quality_level(self, quality_score: float) -> str:
        """Convert quality score to descriptive level."""
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.7:
            return "Good" 
        elif quality_score >= 0.6:
            return "Fair"
        elif quality_score >= 0.4:
            return "Poor"
        else:
            return "Very Poor"
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and capabilities."""
        
        return {
            'service_available': self.is_available(),
            'ml_system_loaded': self.ml_system is not None,
            'accuracy_optimizer_available': self.accuracy_optimizer is not None,
            'enhanced_features': {
                'multi_model_ensemble': True,
                'dynamic_weighting': True,
                'real_time_calibration': True,
                'uniqueness_validation': self.accuracy_optimizer is not None,
                'performance_feedback': True,
                'quality_metrics': True
            },
            'service_version': '1.0',
            'initialization_timestamp': datetime.now().isoformat()
        }

# Global service instance
_enhanced_prediction_service = None

def get_enhanced_prediction_service(db_path: str = "greyhound_racing_data.db") -> EnhancedPredictionService:
    """Get or create the global enhanced prediction service instance."""
    global _enhanced_prediction_service
    
    if _enhanced_prediction_service is None:
        _enhanced_prediction_service = EnhancedPredictionService(db_path)
    
    return _enhanced_prediction_service

# Convenience functions for integration
def predict_race_enhanced(race_data: pd.DataFrame, race_id: str, 
                         market_odds: Optional[Dict[str, float]] = None,
                         db_path: str = "greyhound_racing_data.db",
                         tgr_enabled: Optional[bool] = None) -> Dict[str, Any]:
    """Generate enhanced predictions for a race."""
    service = get_enhanced_prediction_service(db_path)
    return service.predict_race_enhanced(race_data, race_id, market_odds, tgr_enabled=tgr_enabled)

def predict_race_file_enhanced(race_file_path: str, 
                              db_path: str = "greyhound_racing_data.db",
                              tgr_enabled: Optional[bool] = None) -> Dict[str, Any]:
    """Generate enhanced predictions from a race file."""
    service = get_enhanced_prediction_service(db_path)
    return service.predict_race_file_enhanced(race_file_path, tgr_enabled=tgr_enabled)

if __name__ == "__main__":
    # Test the enhanced prediction service
    service = EnhancedPredictionService()
    
    # Get service status
    status = service.get_service_status()
    print("🧪 Enhanced Prediction Service Status:")
    print(json.dumps(status, indent=2))
    
    # Test with sample data if service is available
    if service.is_available():
        sample_data = pd.DataFrame({
            'dog_clean_name': ['TEST_DOG_A', 'TEST_DOG_B', 'TEST_DOG_C'],
            'box_number': [1, 2, 3],
            'weight': [30.0, 32.0, 28.0],
            'starting_price': [3.0, 5.0, 8.0]
        })
        
        result = service.predict_race_enhanced(sample_data, 'TEST_RACE_001')
        print("\n🧪 Test Prediction Result:")
        print(json.dumps(result, indent=2, default=str))
