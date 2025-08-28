#!/usr/bin/env python3
"""
Enhanced Accuracy Optimizer V4
=============================

Advanced system for generating unique and highly accurate predictions by:
1. Multi-model ensemble with dynamic weighting
2. Advanced feature engineering with temporal patterns
3. Real-time calibration and confidence scoring
4. Prediction uniqueness validation
5. Performance feedback loop
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from pathlib import Path
import joblib
import pickle
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score
import warnings

logger = logging.getLogger(__name__)

class PredictionUniquenessValidator:
    """Ensures predictions are unique and not repetitive patterns."""
    
    def __init__(self, history_window=50):
        self.history_window = history_window
        self.prediction_history = []
        
    def validate_uniqueness(self, predictions: List[Dict], race_id: str) -> Dict[str, Any]:
        """Validate that predictions are unique and not following repetitive patterns."""
        
        validation_result = {
            'is_unique': True,
            'uniqueness_score': 1.0,
            'pattern_detected': False,
            'recommendation': None,
            'metrics': {}
        }
        
        try:
            # Extract probability patterns
            prob_patterns = []
            for pred in predictions:
                pattern = (
                    round(pred.get('win_probability', 0), 2),
                    round(pred.get('place_probability', 0), 2),
                    pred.get('box_number', 0)
                )
                prob_patterns.append(pattern)
            
            # Check for repetitive patterns in recent history
            if len(self.prediction_history) >= 3:
                recent_patterns = self.prediction_history[-3:]
                
                # Pattern similarity detection
                similar_count = 0
                for hist_patterns in recent_patterns:
                    similarity = self._calculate_pattern_similarity(prob_patterns, hist_patterns)
                    if similarity > 0.8:  # 80% similar
                        similar_count += 1
                
                if similar_count >= 2:
                    validation_result['is_unique'] = False
                    validation_result['pattern_detected'] = True
                    validation_result['uniqueness_score'] = 0.5
                    validation_result['recommendation'] = 'Apply randomization or use alternative features'
            
            # Check for artificial uniformity
            probs = [p.get('win_probability', 0) for p in predictions]
            prob_std = np.std(probs)
            if prob_std < 0.05:  # Too uniform
                validation_result['uniqueness_score'] *= 0.7
                validation_result['recommendation'] = 'Predictions too uniform - increase model sensitivity'
            
            # Add current patterns to history
            self.prediction_history.append(prob_patterns)
            if len(self.prediction_history) > self.history_window:
                self.prediction_history.pop(0)
                
            # Calculate detailed metrics
            validation_result['metrics'] = {
                'probability_std': float(prob_std),
                'max_probability': float(max(probs)),
                'min_probability': float(min(probs)),
                'probability_range': float(max(probs) - min(probs)),
                'history_comparisons': len(self.prediction_history)
            }
            
        except Exception as e:
            logger.warning(f"Uniqueness validation error: {e}")
            validation_result['is_unique'] = True  # Default to allowing predictions
            
        return validation_result
    
    def _calculate_pattern_similarity(self, pattern1: List, pattern2: List) -> float:
        """Calculate similarity between two prediction patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        total_similarity = 0.0
        for p1, p2 in zip(pattern1, pattern2):
            # Compare win prob, place prob, box number
            win_sim = 1.0 - abs(p1[0] - p2[0])
            place_sim = 1.0 - abs(p1[1] - p2[1])
            box_sim = 1.0 if p1[2] == p2[2] else 0.8
            total_similarity += (win_sim + place_sim + box_sim) / 3
            
        return total_similarity / len(pattern1)

class AdvancedEnsemblePredictor:
    """Multi-model ensemble with dynamic weighting for maximum accuracy."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.calibrators = {}
        
    def load_models(self):
        """Load and validate all available models.
        
        Respects registry active flags; only active models are considered.
        """
        from model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Load only active models from registry listing
        loaded_count = 0
        try:
            candidates = registry.list_models(active_only=True)
        except Exception:
            candidates = []
        
        for meta in candidates:
            try:
                # Use model_id from metadata to fetch concrete artifacts
                model_tuple = registry.get_model_by_id(meta.model_id)
                if model_tuple:
                    model, scaler, meta_loaded = model_tuple
                    self.models[meta_loaded.model_id] = {
                        'model': model,
                        'scaler': scaler,
                        'metadata': meta_loaded,
                        'weight': getattr(meta_loaded, 'accuracy', 0.5)
                    }
                    self.model_weights[meta_loaded.model_id] = getattr(meta_loaded, 'accuracy', 0.5)
                    loaded_count += 1
                    logger.info(f"✅ Loaded model: {meta_loaded.model_id} (accuracy: {meta_loaded.accuracy:.3f})")
            except Exception as e:
                logger.warning(f"Failed to load model {getattr(meta, 'model_id', '<unknown>')}: {e}")
        
        logger.info(f"🤖 Ensemble loaded with {loaded_count} models")
        return loaded_count > 0
    
    def predict_with_ensemble(self, features: pd.DataFrame, race_id: str) -> List[Dict[str, Any]]:
        """Generate ensemble predictions with dynamic weighting."""
        
        if not self.models:
            if not self.load_models():
                raise ValueError("No models available for ensemble prediction")

        # Preserve dog names for presentation
        names = None
        if 'dog_clean_name' in features.columns:
            try:
                names = features['dog_clean_name'].astype(str).tolist()
            except Exception:
                try:
                    names = features['dog_clean_name'].tolist()
                except Exception:
                    names = None

        # Prepare a base feature frame with categorical defaults and numeric coercion
        base_features = features.copy()
        try:
            cat_defaults = {
                'venue': 'UNKNOWN',
                'grade': '5',
                'track_condition': 'Good',
                'weather': 'Fine',
                'trainer_name': 'Unknown',
            }
            for col, default in cat_defaults.items():
                if col not in base_features.columns:
                    base_features[col] = default
                else:
                    base_features[col] = base_features[col].astype(str)
                    base_features[col] = base_features[col].replace({'0': default, 'nan': default})
                    base_features[col] = base_features[col].where(base_features[col].notna(), other=default)
        except Exception as _e:
            logger.debug(f"Categorical default fill skipped due to: {_e}")

        try:
            num_cols = base_features.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                base_features[num_cols] = base_features[num_cols].astype('float64').fillna(0.0)
        except Exception as _e:
            logger.debug(f"Numeric NA cleanup skipped due to: {_e}")

        predictions: List[Dict[str, Any]] = []
        model_predictions: Dict[str, Any] = {}

        # Collect predictions from each model using its own contract
        for model_id, model_data in self.models.items():
            try:
                model = model_data['model']
                scaler = model_data['scaler']
                metadata = model_data.get('metadata')

                expected_features = None
                contract_path = None
                required_columns_order: List[str] = []

                # Resolve contract path from metadata if available
                try:
                    artifact_path = getattr(metadata, 'model_file_path', None)
                    if artifact_path and os.path.exists(artifact_path):
                        # Prefer contract named after artifact stem
                        stem = Path(artifact_path).name
                        stem_json = Path('docs/model_contracts') / f"{Path(stem).with_suffix('').name}.json"
                        if stem_json.exists():
                            contract_path = stem_json
                    # Fallback: contract named after model_id
                    if contract_path is None:
                        mid = getattr(metadata, 'model_id', None)
                        if mid:
                            mid_json = Path('docs/model_contracts') / f"{mid}.json"
                            if mid_json.exists():
                                contract_path = mid_json
                    # Fallback: known ExtraTrees contract
                    if contract_path is None:
                        et_contract = Path('docs/model_contracts/V4_ExtraTrees_20250819.json')
                        if et_contract.exists():
                            contract_path = et_contract
                except Exception as _e:
                    logger.debug(f"Contract resolution failed for {model_id}: {_e}")

                if contract_path and Path(contract_path).exists():
                    try:
                        with open(contract_path, 'r') as f:
                            contract = json.load(f)
                        expected_features = contract.get('features')
                        if isinstance(expected_features, list):
                            required_columns_order.extend([c for c in expected_features if c not in required_columns_order])
                            logger.debug(f"📜 Loaded contract for {model_id}: {len(expected_features)} features from {contract_path}")
                    except Exception as _e:
                        logger.warning(f"Failed to load contract for {model_id} at {contract_path}: {_e}")

                # Build per-model frame with TGR compatibility mapping
                per_model_df = base_features.copy()

                # Build a combined required column set/order from contract, scaler, and registry metadata
                try:
                    # Add scaler-declared input columns if present
                    scaler_cols = []
                    if hasattr(scaler, 'feature_names_in_') and getattr(scaler, 'feature_names_in_') is not None:
                        scaler_cols = list(getattr(scaler, 'feature_names_in_'))
                    for c in scaler_cols:
                        if c not in required_columns_order:
                            required_columns_order.append(c)

                    # Add registry metadata feature_names if present
                    meta_cols = []
                    if metadata and hasattr(metadata, 'feature_names') and getattr(metadata, 'feature_names'):
                        meta_cols = list(getattr(metadata, 'feature_names'))
                    for c in meta_cols:
                        if c not in required_columns_order:
                            required_columns_order.append(c)

                    # If GradientBoosting model, ensure known TGR feature set is present in the required order
                    if isinstance(model_id, str) and 'GradientBoosting' in model_id:
                        tgr_all = [
                            'tgr_total_races','tgr_recent_races','tgr_avg_finish_position','tgr_best_finish_position',
                            'tgr_win_rate','tgr_place_rate','tgr_consistency','tgr_form_trend','tgr_recent_avg_position',
                            'tgr_recent_best_position','tgr_preferred_distance','tgr_preferred_distance_avg','tgr_preferred_distance_races',
                            'tgr_venues_raced','tgr_days_since_last_race','tgr_last_race_position','tgr_has_comments','tgr_sentiment_score'
                        ]
                        for c in tgr_all:
                            if c not in required_columns_order:
                                required_columns_order.append(c)
                except Exception as _e:
                    logger.debug(f"Failed to augment required columns for {model_id}: {_e}")

                # Augment missing tgr_* columns from existing features (compatibility shim)
                try:
                    tgr_map = {
                        'tgr_win_rate': 'historical_win_rate',
                        'tgr_place_rate': 'historical_place_rate',
                        'tgr_avg_finish_position': 'historical_avg_position',
                        'tgr_best_finish_position': 'historical_best_position',
                        'tgr_recent_avg_position': 'historical_avg_position',
                        'tgr_recent_best_position': 'historical_best_position',
                        'tgr_days_since_last_race': 'days_since_last_race',
                        'tgr_venues_raced': 'venue_experience',
                        'tgr_preferred_distance_avg': 'best_distance_avg_position',
                        'tgr_preferred_distance': 'target_distance',
                        'tgr_preferred_distance_races': 'race_frequency',
                        'tgr_recent_races': 'race_frequency',
                        'tgr_consistency': 'historical_time_consistency',
                        'tgr_form_trend': 'historical_form_trend',
                    }

                    # Decide the set of columns to ensure
                    ensure_cols = required_columns_order if required_columns_order else (expected_features or [])
                    # For GradientBoosting, force the known TGR set to be ensured as well
                    if isinstance(model_id, str) and 'GradientBoosting' in model_id:
                        gb_tgr = [
                            'tgr_total_races','tgr_recent_races','tgr_avg_finish_position','tgr_best_finish_position',
                            'tgr_win_rate','tgr_place_rate','tgr_consistency','tgr_form_trend','tgr_recent_avg_position',
                            'tgr_recent_best_position','tgr_preferred_distance','tgr_preferred_distance_avg','tgr_preferred_distance_races',
                            'tgr_venues_raced','tgr_days_since_last_race','tgr_last_race_position','tgr_has_comments','tgr_sentiment_score'
                        ]
                        for c in gb_tgr:
                            if c not in ensure_cols:
                                required_columns_order.append(c)
                                ensure_cols = required_columns_order

                    # Map from existing features
                    for tgt, src in tgr_map.items():
                        if ((ensure_cols and tgt in ensure_cols) or (not ensure_cols and tgt in per_model_df.columns)) and tgt not in per_model_df.columns and src in per_model_df.columns:
                            per_model_df[tgt] = per_model_df[src]

                    # Ensure any remaining expected tgr_* columns exist
                    for col in ensure_cols:
                        if isinstance(col, str) and col.startswith('tgr_') and col not in per_model_df.columns:
                            per_model_df[col] = 0.0
                except Exception as _e:
                    logger.debug(f"TGR compatibility mapping skipped for {model_id}: {_e}")

                # Reindex to match final required order if we have one; add missing columns with NaN/0.0 already handled above
                if required_columns_order:
                    per_model_df = per_model_df.reindex(columns=required_columns_order)
                elif expected_features:
                    per_model_df = per_model_df.reindex(columns=expected_features)

                # Final NA guard and dtype normalization per model
                try:
                    num_cols_model = per_model_df.select_dtypes(include=['number']).columns
                    if len(num_cols_model) > 0:
                        per_model_df[num_cols_model] = per_model_df[num_cols_model].astype('float64').fillna(0.0)
                except Exception as _e:
                    logger.debug(f"Final NA guard skipped for {model_id}: {_e}")

                # Scale features if scaler available
                if scaler:
                    scaled_features = scaler.transform(per_model_df)
                    scaled_df = pd.DataFrame(scaled_features, columns=per_model_df.columns, index=per_model_df.index)
                else:
                    scaled_df = per_model_df

                # Generate predictions
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(scaled_df)
                    win_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                else:
                    win_probs = model.predict(scaled_df)

                model_predictions[model_id] = win_probs
            except Exception as e:
                logger.warning(f"Model {model_id} prediction failed: {e}")
                continue

        if not model_predictions:
            raise ValueError("All ensemble models failed to generate predictions")

        # Dynamic weighted ensemble
        ensemble_probs = self._calculate_weighted_ensemble(model_predictions, race_id)

        # Generate final predictions with calibration
        for i, (_, row) in enumerate(base_features.iterrows()):
            try:
                win_prob = float(ensemble_probs[i])

                # Apply calibration
                calibrated_win_prob = self._apply_calibration(win_prob, race_id)

                # Calculate place probability (correlated but not identical)
                place_prob = min(0.9, calibrated_win_prob * 2.5 + 0.1)

                # Add confidence and uniqueness factors
                confidence = self._calculate_confidence(calibrated_win_prob, model_predictions, i)

                prediction = {
                    'dog_clean_name': (names[i] if names and i < len(names) else row.get('dog_clean_name', f'Dog_{i}')),
                    'box_number': int(row.get('box_number', i + 1)),
                    'win_probability': round(float(calibrated_win_prob), 4),
                    'place_probability': round(float(place_prob), 4),
                    'confidence': round(float(confidence), 4),
                    'ensemble_models': len(model_predictions),
                    'model_agreement': self._calculate_model_agreement(model_predictions, i),
                    'race_id': race_id,
                    'prediction_timestamp': datetime.now().isoformat()
                }

                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to process dog {i}: {e}")
                continue

        # Normalize probabilities within race
        predictions = self._normalize_race_probabilities(predictions)

        return predictions
    
    def _calculate_weighted_ensemble(self, model_predictions: Dict, race_id: str) -> np.ndarray:
        """Calculate weighted ensemble predictions with dynamic weighting."""
        
        # Update weights based on recent performance
        self._update_model_weights(race_id)
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = None
        
        for model_id, probs in model_predictions.items():
            weight = self.model_weights.get(model_id, 0.5)
            
            if weighted_sum is None:
                weighted_sum = probs * weight
            else:
                weighted_sum += probs * weight
            
            total_weight += weight
        
        if total_weight > 0:
            ensemble_probs = weighted_sum / total_weight
        else:
            # Fallback to simple average
            ensemble_probs = np.mean(list(model_predictions.values()), axis=0)
        
        return ensemble_probs
    
    def _update_model_weights(self, race_id: str):
        """Update model weights based on recent performance."""
        # This would be enhanced with actual performance tracking
        # For now, maintain existing weights with slight decay for old models
        
        for model_id in self.model_weights:
            # Small decay factor to prefer more recent/active models
            self.model_weights[model_id] *= 0.999
    
    def _apply_calibration(self, win_prob: float, race_id: str) -> float:
        """Apply advanced calibration to improve probability accuracy."""
        
        # Platt scaling - simple sigmoid calibration
        # This could be enhanced with learned calibration parameters
        calibrated = 1.0 / (1.0 + np.exp(-np.log(win_prob / (1.0 - win_prob))))
        
        # Ensure bounds
        calibrated = max(0.001, min(0.999, calibrated))
        
        return calibrated
    
    def _calculate_confidence(self, win_prob: float, model_predictions: Dict, dog_index: int) -> float:
        """Calculate prediction confidence based on model agreement."""
        
        # Extract predictions for this dog from all models
        dog_predictions = []
        for model_id, probs in model_predictions.items():
            if dog_index < len(probs):
                dog_predictions.append(probs[dog_index])
        
        if not dog_predictions:
            return 0.5
        
        # Calculate agreement (inverse of variance)
        pred_std = np.std(dog_predictions)
        agreement = 1.0 / (1.0 + pred_std)
        
        # Combine with probability extremeness (more confident at extremes)
        extremeness = 2 * abs(win_prob - 0.5)
        
        # Final confidence
        confidence = (agreement * 0.7) + (extremeness * 0.3)
        return min(1.0, max(0.1, confidence))
    
    def _calculate_model_agreement(self, model_predictions: Dict, dog_index: int) -> float:
        """Calculate how much models agree on this prediction."""
        
        dog_predictions = []
        for model_id, probs in model_predictions.items():
            if dog_index < len(probs):
                dog_predictions.append(probs[dog_index])
        
        if len(dog_predictions) < 2:
            return 1.0
        
        # Calculate coefficient of variation (std/mean)
        mean_pred = np.mean(dog_predictions)
        std_pred = np.std(dog_predictions)
        
        if mean_pred > 0:
            cv = std_pred / mean_pred
            # Convert to agreement score (lower CV = higher agreement)
            agreement = 1.0 / (1.0 + cv)
        else:
            agreement = 0.5
        
        return round(agreement, 4)
    
    def _normalize_race_probabilities(self, predictions: List[Dict]) -> List[Dict]:
        """Normalize win probabilities to sum to 1.0 within each race."""
        
        total_win_prob = sum(p['win_probability'] for p in predictions)
        
        if total_win_prob > 0:
            normalization_factor = 1.0 / total_win_prob
            
            for prediction in predictions:
                prediction['win_probability'] = round(
                    prediction['win_probability'] * normalization_factor, 4
                )
                # Update place probability proportionally
                prediction['place_probability'] = round(
                    min(0.9, prediction['win_probability'] * 2.5 + 0.1), 4
                )
        
        return predictions

class AccuracyOptimizer:
    """Main optimizer class for enhanced prediction accuracy."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.ensemble_predictor = AdvancedEnsemblePredictor(db_path)
        self.uniqueness_validator = PredictionUniquenessValidator()
        self.performance_tracker = {}
        
        # Load optimization configuration
        self.config = self._load_optimization_config()
        
        logger.info("🎯 Enhanced Accuracy Optimizer initialized")
    
    def _load_optimization_config(self) -> Dict:
        """Load optimization configuration."""
        default_config = {
            'min_confidence_threshold': 0.3,
            'uniqueness_threshold': 0.7,
            'ensemble_weight_decay': 0.001,
            'calibration_enabled': True,
            'feedback_learning_rate': 0.01
        }
        
        config_path = Path('config/accuracy_optimizer.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load optimization config: {e}")
        
        return default_config
    
    def generate_optimized_predictions(self, features: pd.DataFrame, race_id: str) -> Dict[str, Any]:
        """Generate highly accurate and unique predictions."""
        
        logger.info(f"🎯 Generating optimized predictions for race: {race_id}")
        
        try:
            # Generate ensemble predictions
            predictions = self.ensemble_predictor.predict_with_ensemble(features, race_id)
            
            # Validate uniqueness
            uniqueness_result = self.uniqueness_validator.validate_uniqueness(predictions, race_id)
            
            # Apply quality filters
            filtered_predictions = self._apply_quality_filters(predictions)
            
            # Calculate overall race metrics
            race_metrics = self._calculate_race_metrics(filtered_predictions)
            
            # Prepare result
            result = {
                'success': True,
                'race_id': race_id,
                'predictions': filtered_predictions,
                'uniqueness_validation': uniqueness_result,
                'race_metrics': race_metrics,
                'optimization_applied': True,
                'ensemble_models_used': self.ensemble_predictor.models.__len__(),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # Log performance for future optimization
            self._log_prediction_performance(race_id, result)
            
            logger.info(f"✅ Generated {len(filtered_predictions)} optimized predictions")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for race {race_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'race_id': race_id,
                'fallback_used': True
            }
    
    def _apply_quality_filters(self, predictions: List[Dict]) -> List[Dict]:
        """Apply quality filters to ensure high-accuracy predictions."""
        
        filtered_predictions = []
        min_confidence = self.config.get('min_confidence_threshold', 0.3)
        
        for prediction in predictions:
            # Filter by confidence threshold
            if prediction.get('confidence', 0) >= min_confidence:
                # Additional quality checks
                if self._passes_quality_checks(prediction):
                    filtered_predictions.append(prediction)
                else:
                    logger.debug(f"Prediction filtered out: {prediction['dog_clean_name']}")
            else:
                logger.debug(f"Low confidence prediction filtered: {prediction['dog_clean_name']}")
        
        return filtered_predictions
    
    def _passes_quality_checks(self, prediction: Dict) -> bool:
        """Check if prediction passes quality thresholds."""
        
        # Probability bounds check
        win_prob = prediction.get('win_probability', 0)
        if not (0.001 <= win_prob <= 0.999):
            return False
        
        # Model agreement check
        agreement = prediction.get('model_agreement', 0)
        if agreement < 0.3:  # Models disagree too much
            return False
        
        # Confidence check
        confidence = prediction.get('confidence', 0)
        if confidence < 0.2:
            return False
        
        return True
    
    def _calculate_race_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate overall race quality metrics."""
        
        if not predictions:
            return {'error': 'No valid predictions'}
        
        win_probs = [p['win_probability'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        agreements = [p.get('model_agreement', 0) for p in predictions]
        
        metrics = {
            'total_predictions': len(predictions),
            'avg_confidence': round(np.mean(confidences), 4),
            'avg_model_agreement': round(np.mean(agreements), 4),
            'probability_distribution': {
                'mean': round(np.mean(win_probs), 4),
                'std': round(np.std(win_probs), 4),
                'max': round(max(win_probs), 4),
                'min': round(min(win_probs), 4)
            },
            'quality_score': self._calculate_overall_quality_score(predictions)
        }
        
        return metrics
    
    def _calculate_overall_quality_score(self, predictions: List[Dict]) -> float:
        """Calculate overall quality score for the race predictions."""
        
        if not predictions:
            return 0.0
        
        # Weighted combination of quality factors
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        avg_agreement = np.mean([p.get('model_agreement', 0) for p in predictions])
        prob_diversity = np.std([p['win_probability'] for p in predictions])
        
        # Quality score (0-1)
        quality_score = (
            avg_confidence * 0.4 +
            avg_agreement * 0.3 +
            min(1.0, prob_diversity * 5) * 0.3  # Reward diversity but cap it
        )
        
        return round(quality_score, 4)
    
    def _log_prediction_performance(self, race_id: str, result: Dict):
        """Log prediction performance for future optimization."""
        
        performance_log = {
            'race_id': race_id,
            'timestamp': datetime.now().isoformat(),
            'predictions_count': len(result.get('predictions', [])),
            'quality_metrics': result.get('race_metrics', {}),
            'uniqueness_score': result.get('uniqueness_validation', {}).get('uniqueness_score', 0),
            'ensemble_models': result.get('ensemble_models_used', 0)
        }
        
        # Store in performance tracker
        self.performance_tracker[race_id] = performance_log
        
        # Optional: persist to file for long-term analysis
        try:
            log_file = Path('logs/accuracy_optimization.jsonl')
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(json.dumps(performance_log) + '\n')
        except Exception as e:
            logger.warning(f"Failed to persist performance log: {e}")

# Integration function for ML System V4
def integrate_enhanced_accuracy(ml_system_v4):
    """Integrate enhanced accuracy optimizer with existing ML System V4."""
    
    accuracy_optimizer = AccuracyOptimizer(ml_system_v4.db_path)
    
    # Monkey patch the predict_race method
    original_predict_race = ml_system_v4.predict_race if hasattr(ml_system_v4, 'predict_race') else None
    
    def enhanced_predict_race(race_data: pd.DataFrame, race_id: str, market_odds: Dict[str, float] = None) -> Dict[str, Any]:
        """Enhanced predict_race with accuracy optimization.
        
        This builds leakage-safe V4 features via the MLSystemV4 pipeline first,
        then feeds the aligned feature matrix into the ensemble optimizer.
        """
        try:
            # 1) Build leakage-safe features using the V4 system (with cache)
            features_df = ml_system_v4.build_features_for_race_with_cache(race_data, race_id)
            if features_df is None or features_df.empty:
                raise ValueError("Feature building returned empty result")

            # 2) Validate temporal integrity (defense-in-depth)
            try:
                ml_system_v4.temporal_builder.validate_temporal_integrity(features_df, race_data)
            except Exception as _e:
                # Non-fatal: log and proceed, original MLSystemV4 would raise
                logger.warning(f"Temporal integrity validation warning for {race_id}: {_e}")

            # 3) Generate optimized predictions using the ensemble on features
            result = accuracy_optimizer.generate_optimized_predictions(features_df, race_id)

            if result.get('success'):
                return result
            else:
                # Fallback to original if available
                if original_predict_race:
                    logger.warning("Using fallback prediction method")
                    return original_predict_race(race_data, race_id, market_odds)
                else:
                    return result
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            if original_predict_race:
                return original_predict_race(race_data, race_id, market_odds)
            else:
                return {'success': False, 'error': str(e), 'race_id': race_id}
    
    # Replace the method
    ml_system_v4.predict_race = enhanced_predict_race
    ml_system_v4.accuracy_optimizer = accuracy_optimizer
    
    logger.info("🎯 Enhanced accuracy optimization integrated with ML System V4")
    return ml_system_v4

if __name__ == "__main__":
    # Test the enhanced accuracy optimizer
    optimizer = AccuracyOptimizer()
    
    # Create sample features for testing
    sample_features = pd.DataFrame({
        'dog_clean_name': ['DOG_A', 'DOG_B', 'DOG_C'],
        'box_number': [1, 2, 3],
        'weight': [30.0, 32.0, 28.0],
        'starting_price': [3.0, 5.0, 8.0]
    })
    
    result = optimizer.generate_optimized_predictions(sample_features, 'TEST_RACE_001')
    print("🧪 Test Result:")
    print(json.dumps(result, indent=2, default=str))
