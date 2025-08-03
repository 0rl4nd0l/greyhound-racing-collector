#!/usr/bin/env python3
"""
ML System V4 - Temporal Leakage-Safe with Calibration & EV
==========================================================

Features:
- Temporal leakage protection via TemporalFeatureBuilder
- Proper sklearn pipeline with ColumnTransformer
- ExtraTreesClassifier with CalibratedClassifierCV
- Group-normalized probabilities (softmax per race)
- Expected Value (EV) calculation
- Time-ordered train/test splits
- Comprehensive testing and validation
"""

import logging
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Sklearn imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.inspection import permutation_importance
import joblib

# Import our temporal feature builder
from temporal_feature_builder import TemporalFeatureBuilder, create_temporal_assertion_hook

# Import profiling infrastructure
from pipeline_profiler import profile_function, track_sequence, pipeline_profiler

logger = logging.getLogger(__name__)

class MLSystemV4:
    """Temporal leakage-safe ML system with calibration and EV calculation."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.temporal_builder = TemporalFeatureBuilder(db_path)
        self.pipeline = None
        self.calibrated_pipeline = None
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.model_info = {}
        self.ev_thresholds = {}

        # Adjust for upcoming races
        self.upcoming_race_box_numbers = True
        self.upcoming_race_weights = True
        self.disable_odds = True
        
        # Create temporal assertion hook
        self.assert_no_leakage = create_temporal_assertion_hook()
        
        # Try to load existing model
        self._try_load_latest_model()
        
        logger.info("🛡️ ML System V4 initialized with temporal leakage protection")
    
    def prepare_time_ordered_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data with time-ordered splits (no random splits)."""
        logger.info("📅 Preparing time-ordered training data...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load data with temporal information
            query = """
            SELECT 
                d.*,
                r.venue, r.grade, r.distance, r.track_condition, r.weather,
                r.temperature, r.humidity, r.wind_speed, r.field_size,
                r.race_date, r.race_time, r.winner_name, r.winner_odds, r.winner_margin,
                e.pir_rating, e.first_sectional, e.win_time, e.bonus_time
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            LEFT JOIN enhanced_expert_data e ON d.race_id = e.race_id 
                AND d.dog_clean_name = e.dog_clean_name
            WHERE d.race_id IS NOT NULL 
                AND r.race_date IS NOT NULL
                AND d.finish_position IS NOT NULL
            ORDER BY r.race_date ASC, r.race_time ASC, d.race_id, d.box_number
            """
            
            raw_data = pd.read_sql_query(query, conn)
            conn.close()
            
            if raw_data.empty:
                logger.error("No data available for training")
                return pd.DataFrame(), pd.DataFrame()
            
            # Parse timestamps
            raw_data['race_timestamp'] = raw_data.apply(
                lambda row: self.temporal_builder.get_race_timestamp(row), axis=1
            )
            
            # Group by race_id to maintain race integrity
            race_groups = raw_data.groupby('race_id')
            race_timestamps = race_groups['race_timestamp'].first().sort_values()
            
            # Time-ordered split (80/20)
            split_point = int(len(race_timestamps) * 0.8)
            train_race_ids = set(race_timestamps.iloc[:split_point].index)
            test_race_ids = set(race_timestamps.iloc[split_point:].index)
            
            # Ensure no race_id appears in both train and test
            assert len(train_race_ids.intersection(test_race_ids)) == 0, "Race ID overlap detected!"
            
            train_data = raw_data[raw_data['race_id'].isin(train_race_ids)].copy()
            test_data = raw_data[raw_data['race_id'].isin(test_race_ids)].copy()
            
            logger.info(f"📊 Time-ordered split:")
            logger.info(f"   Training: {len(train_race_ids)} races, {len(train_data)} samples")
            logger.info(f"   Testing:  {len(test_race_ids)} races, {len(test_data)} samples")
            logger.info(f"   Train period: {train_data['race_timestamp'].min()} to {train_data['race_timestamp'].max()}")
            logger.info(f"   Test period:  {test_data['race_timestamp'].min()} to {test_data['race_timestamp'].max()}")
            
            return train_data, test_data
        
        except Exception as e:
            logger.error(f"Error preparing time-ordered data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    @profile_function
    def build_leakage_safe_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Build features using temporal feature builder to prevent leakage."""
        with track_sequence('feature_engineering', 'MLSystemV4', 'feature_engineering'):
            logger.info("🔧 Building leakage-safe features...")
            
            all_features = []
            
            # Process each race separately
            for race_id in raw_data['race_id'].unique():
                race_data = raw_data[raw_data['race_id'] == race_id]
                
                # Build temporal leakage-safe features
                if self.upcoming_race_box_numbers:
                    race_data['box_number'] = range(1, len(race_data) + 1)

                if self.upcoming_race_weights and 'weight' in race_data.columns:
                    # Weight column already exists, no need to map
                    pass

                race_features = self.temporal_builder.build_features_for_race(race_data, race_id)

                # Validate temporal integrity
                self.temporal_builder.validate_temporal_integrity(race_features, race_data)
                
                all_features.append(race_features)
            
            # Combine all race features
            combined_features = pd.concat(all_features, ignore_index=True)
            
            logger.info(f"✅ Built {len(combined_features)} feature vectors across {len(raw_data['race_id'].unique())} races")
            
            return combined_features
    
    def create_sklearn_pipeline(self, features_df: pd.DataFrame) -> Pipeline:
        """Create sklearn pipeline with proper encoding and calibration."""
        logger.info("🏗️ Creating sklearn pipeline with calibration...")
        
        # Identify feature types
        self.categorical_columns = [
            col for col in features_df.columns 
            if col in ['venue', 'grade', 'track_condition', 'weather', 'trainer_name']
        ]
        
        # Exclude metadata columns and non-numeric columns from training
        exclude_columns = ['race_id', 'dog_clean_name', 'target', 'target_timestamp', 'race_date', 'race_time']
        
        self.numerical_columns = [
            col for col in features_df.columns 
            if col not in self.categorical_columns + exclude_columns and 
               pd.api.types.is_numeric_dtype(features_df[col])
        ]
        
        logger.info(f"📊 Feature composition:")
        logger.info(f"   Categorical: {len(self.categorical_columns)} features")
        logger.info(f"   Numerical: {len(self.numerical_columns)} features")
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_columns)
            ],
            remainder='drop'
        )
        
        # Create base model with proper hyperparameters
        base_model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_model)
        ])
        
        # Wrap with calibration
        calibrated_pipeline = CalibratedClassifierCV(
            pipeline, 
            method='isotonic',  # Preferred over sigmoid
            cv=3
        )
        
        return calibrated_pipeline
    
    def train_model(self) -> bool:
        """Train the complete leakage-safe, calibrated model."""
        logger.info("🚀 Starting leakage-safe model training...")
        
        # Prepare time-ordered data
        train_data, test_data = self.prepare_time_ordered_data()
        if train_data.empty or test_data.empty:
            logger.error("No data available for training")
            return False
        
        # Build leakage-safe features
        train_features = self.build_leakage_safe_features(train_data)
        test_features = self.build_leakage_safe_features(test_data)
        
        if train_features.empty or test_features.empty:
            logger.error("No features created")
            return False
        
        # Prepare training data
        X_train = train_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
        y_train = train_features['target']
        X_test = test_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
        y_test = test_features['target']
        
        # Store feature columns for later use
        self.feature_columns = X_train.columns.tolist()
        
        logger.info(f"📊 Training data shape: {X_train.shape}")
        logger.info(f"📊 Test data shape: {X_test.shape}")
        logger.info(f"🎯 Target distribution - Train wins: {y_train.sum()}/{len(y_train)} ({y_train.mean():.3f})")
        logger.info(f"🎯 Target distribution - Test wins: {y_test.sum()}/{len(y_test)} ({y_test.mean():.3f})")
        
        # Create and train pipeline
        self.calibrated_pipeline = self.create_sklearn_pipeline(X_train)
        
        # Train the model
        logger.info("🎯 Training calibrated model...")
        self.calibrated_pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_pred_proba = self.calibrated_pipeline.predict_proba(X_train)[:, 1]
        test_pred_proba = self.calibrated_pipeline.predict_proba(X_test)[:, 1]
        
        train_pred = (train_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        train_brier = brier_score_loss(y_train, train_pred_proba)
        test_brier = brier_score_loss(y_test, test_pred_proba)
        
        logger.info("📈 Model Performance:")
        logger.info(f"   Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"   Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"   Training AUC: {train_auc:.4f}")
        logger.info(f"   Test AUC: {test_auc:.4f}")
        logger.info(f"   Training Brier Score: {train_brier:.4f}")
        logger.info(f"   Test Brier Score: {test_brier:.4f}")
        
        # Feature importance
        try:
            # Get feature names after preprocessing
            feature_names = self._get_feature_names_after_preprocessing()
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.calibrated_pipeline, X_test, y_test, 
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            # Log top features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean
            }).sort_values('importance', ascending=False)
            
            logger.info("🔍 Top 10 Feature Importances:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
        
        # Learn EV thresholds
        self.ev_thresholds = self._learn_ev_thresholds(test_features, test_pred_proba)
        
        # Save model info
        self.model_info = {
            'model_type': 'ExtraTreesClassifier_Calibrated',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_brier': train_brier,
            'test_brier': test_brier,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'temporal_split': True,
            'calibration_method': 'isotonic',
            'ev_thresholds': self.ev_thresholds,
            'trained_at': datetime.now().isoformat()
        }
        
        # Save model
        model_path = self._save_model()
        logger.info(f"✅ Model training completed! Saved to {model_path}")
        
        return True
    
    def predict_race(self, race_data: pd.DataFrame, race_id: str, 
                    market_odds: Dict[str, float] = None) -> Dict[str, Any]:
        """Make predictions for all dogs in a race with group normalization and EV."""
        if not self.calibrated_pipeline:
            logger.error("No calibrated model available for prediction")
            return {'success': False, 'error': 'No model loaded'}
        
        try:
            # Build leakage-safe features for the race
            race_features = self.temporal_builder.build_features_for_race(race_data, race_id)
            
            # Validate temporal integrity (critical assertion)
            self.temporal_builder.validate_temporal_integrity(race_features, race_data)
            
            # Prepare features for prediction
            X_pred = race_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], 
                                       axis=1, errors='ignore')
            
            # Ensure all required columns are present
            X_pred = X_pred.reindex(columns=self.feature_columns, fill_value=0)
            
            # Assert no temporal leakage at prediction time
            for idx, row in race_features.iterrows():
                dog_name = row['dog_clean_name']
                features_dict = X_pred.iloc[idx].to_dict()
                self.assert_no_leakage(features_dict, race_id, dog_name)
            
            # Make raw predictions with instrumentation (inference)
            with track_sequence('inference', 'MLSystemV4', 'inference'):
                calibration_present = 'calibration_meta' in self.model_info
                predict_proba_lambda = lambda x: self.calibrated_pipeline.predict_proba(x)[:, 1]
                raw_win_probabilities = predict_proba_lambda(X_pred)
                
                # Record calibration stage
                if calibration_present:
                    with track_sequence('calibration', 'MLSystemV4', 'calibration'):
                        logger.info('Recording calibration stage after predict_proba as calibration is present.')
            
            # Group normalization (softmax within race)
            normalized_win_probs = self._group_normalize_probabilities(raw_win_probabilities)
            
            # Validate normalization
            prob_sum = np.sum(normalized_win_probs)
            if not (0.95 <= prob_sum <= 1.05):
                logger.warning(f"Normalization check failed: sum = {prob_sum:.4f}")
            
            # Calculate place probabilities (heuristic)
            normalized_place_probs = np.minimum(0.95, normalized_win_probs * 2.8)
            
            # Calculate EV if market odds available
            ev_calculations = {}
            if market_odds:
                for i, dog_name in enumerate(race_features['dog_clean_name']):
                    if dog_name in market_odds:
                        odds = market_odds[dog_name]
                        win_prob = normalized_win_probs[i]
                        ev_win = win_prob * (odds - 1) - (1 - win_prob)
                        ev_calculations[dog_name] = {
                            'odds': odds,
                            'ev_win': ev_win,
                            'ev_positive': ev_win > 0
                        }
            
            # Create prediction results
            predictions = []
            for i, row in race_features.iterrows():
                dog_name = row['dog_clean_name']
                
                prediction = {
                    'dog_name': dog_name,
                    'box_number': row.get('box_number', i + 1),
                    'win_prob_raw': float(raw_win_probabilities[i]),
                    'win_prob_norm': float(normalized_win_probs[i]),
                    'place_prob_norm': float(normalized_place_probs[i]),
                    'confidence': self._calculate_prediction_confidence(X_pred.iloc[i]),
                    'calibration_applied': True
                }
                
                # Add EV if available
                if dog_name in ev_calculations:
                    prediction.update(ev_calculations[dog_name])
                
                predictions.append(prediction)
            
            # Sort by normalized win probability
            predictions.sort(key=lambda x: x['win_prob_norm'], reverse=True)
            
            # Add ranking
            for i, pred in enumerate(predictions):
                pred['predicted_rank'] = i + 1
            
            # Calculate explainability metadata
            explainability_meta = self._create_explainability_metadata(X_pred, race_id)
            
            result = {
                'success': True,
                'race_id': race_id,
                'predictions': predictions,
                'model_info': self.model_info.get('model_type', 'unknown'),
                'calibration_meta': {
                    'method': 'isotonic',
                    'applied': True,
                    'normalization_sum': float(prob_sum)
                },
                'explainability_meta': explainability_meta,
                'ev_meta': {
                    'thresholds': self.ev_thresholds,
                    'calculations_available': len(ev_calculations) > 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Race prediction complete for {race_id}: {len(predictions)} dogs")
            
            # Expose internal profiler flush
            self.flush_profiler()
            
            return result
        
        except Exception as e:
            logger.error(f"Error predicting race {race_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'race_id': race_id,
                'fallback_reason': f'Prediction pipeline error: {str(e)}'
            }
    
    @profile_function
    def _group_normalize_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply softmax normalization to ensure probabilities sum to 1."""
        with track_sequence('post', 'MLSystemV4', 'post'):
            # Softmax normalization
            exp_probs = np.exp(probabilities - np.max(probabilities))  # Subtract max for numerical stability
            normalized = exp_probs / np.sum(exp_probs)
            return normalized
    
    def _learn_ev_thresholds(self, test_features: pd.DataFrame, 
                           test_probabilities: np.ndarray) -> Dict[str, float]:
        """Learn optimal EV thresholds based on ROI optimization."""
        logger.info("📊 Learning EV thresholds...")
        
        # Simulate market odds (would be replaced with real odds in production)
        simulated_odds = 1.0 / (test_probabilities + 0.01)  # Inverse probability with small epsilon
        
        # Test different EV thresholds
        thresholds = np.arange(0.0, 0.15, 0.01)
        best_roi = -float('inf')
        best_threshold = 0.05
        
        for threshold in thresholds:
            # Calculate EV for each prediction
            ev_values = test_probabilities * (simulated_odds - 1) - (1 - test_probabilities)
            
            # Select bets above threshold
            bet_mask = ev_values > threshold
            
            if np.sum(bet_mask) == 0:
                continue
            
            # Calculate ROI (simplified)
            actual_outcomes = test_features['target'].values
            bet_outcomes = actual_outcomes[bet_mask]
            bet_odds = simulated_odds[bet_mask]
            
            # Calculate returns
            returns = np.where(bet_outcomes == 1, bet_odds - 1, -1)
            roi = np.mean(returns) if len(returns) > 0 else -1
            
            if roi > best_roi:
                best_roi = roi
                best_threshold = threshold
        
        thresholds_dict = {
            'win_threshold': float(best_threshold),
            'place_threshold': float(best_threshold * 0.7),  # Lower threshold for place bets
            'quinella_threshold': float(best_threshold * 1.2),  # Higher threshold for exotic bets
            'learned_roi': float(best_roi)
        }
        
        logger.info(f"📈 Learned EV thresholds: {thresholds_dict}")
        return thresholds_dict
    
    def _calculate_prediction_confidence(self, features: pd.Series) -> float:
        """Calculate confidence based on feature completeness and model certainty."""
        # Feature completeness
        non_zero_features = np.sum(features != 0)
        total_features = len(features)
        completeness = non_zero_features / total_features
        
        # Return scaled confidence
        return min(0.95, completeness * 0.8 + 0.2)
    
    def _create_explainability_metadata(self, X_pred: pd.DataFrame, race_id: str) -> Dict[str, Any]:
        """Create explainability metadata for logging."""
        try:
            # Calculate mean feature values for this race
            feature_summary = {
                'mean_values': X_pred.mean().to_dict(),
                'feature_count': len(X_pred.columns),
                'samples_count': len(X_pred),
                'log_path': f'logs/explainability_{race_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            }
            
            # Save detailed explainability to logs
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)
            
            with open(logs_dir / f'explainability_{race_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump({
                    'race_id': race_id,
                    'timestamp': datetime.now().isoformat(),
                    'feature_values': X_pred.to_dict('records'),
                    'feature_names': X_pred.columns.tolist()
                }, f, indent=2)
            
            return feature_summary
        
        except Exception as e:
            logger.warning(f"Could not create explainability metadata: {e}")
            return {'error': str(e)}
    
    def _get_feature_names_after_preprocessing(self) -> List[str]:
        """Get feature names after preprocessing pipeline."""
        try:
            # Get the fitted preprocessor
            preprocessor = self.calibrated_pipeline.base_estimator_.named_steps['preprocessor']
            
            # Numerical features (passthrough)
            num_features = self.numerical_columns
            
            # Categorical features (after one-hot encoding)
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_features = []
            for i, col in enumerate(self.categorical_columns):
                if hasattr(cat_encoder, 'categories_'):
                    categories = cat_encoder.categories_[i]
                    cat_features.extend([f"{col}_{cat}" for cat in categories])
            
            return num_features + cat_features
        
        except Exception as e:
            logger.warning(f"Could not get feature names: {e}")
            return self.feature_columns
    
    def _save_model(self) -> Path:
        """Save the complete model pipeline."""
        model_dir = Path('./ml_models_v4')
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'ml_model_v4_{timestamp}.joblib'
        
        model_data = {
            'calibrated_pipeline': self.calibrated_pipeline,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'model_info': self.model_info,
            'ev_thresholds': self.ev_thresholds,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"💾 Model saved to {model_path}")
        
        return model_path
    
    @profile_function
    def _try_load_latest_model(self):
        """Try to load the latest model."""
        with track_sequence('model_load', 'MLSystemV4', 'model_load'):
            model_dir = Path('./ml_models_v4')
            if not model_dir.exists():
                logger.info("No model directory found")
                return
            
            model_files = list(model_dir.glob('ml_model_v4_*.joblib'))
            if not model_files:
                logger.info("No trained models found")
                return
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            try:
                model_data = joblib.load(latest_model)
                self.calibrated_pipeline = model_data.get('calibrated_pipeline')
                self.feature_columns = model_data.get('feature_columns', [])
                self.categorical_columns = model_data.get('categorical_columns', [])
                self.numerical_columns = model_data.get('numerical_columns', [])
                self.model_info = model_data.get('model_info', {})
                self.ev_thresholds = model_data.get('ev_thresholds', {})
                
                logger.info(f"📥 Loaded model from {latest_model}")
                logger.info(f"📊 Model info: {self.model_info.get('model_type', 'unknown')}")
            
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.calibrated_pipeline = None


# Training function
def train_leakage_safe_model():
    """Train a new leakage-safe model."""
    system = MLSystemV4()
    success = system.train_model()
    
    if success:
        return {
            'success': True,
            'message': 'Leakage-safe model trained successfully',
            'model_info': system.model_info
        }
    else:
        return {
            'success': False,
            'message': 'Model training failed'
        }


if __name__ == "__main__":
    # Test the system
    logger.info("🧪 Testing ML System V4...")
    
    system = MLSystemV4()
    
    # Train model
    success = system.train_model()
    
    if success:
        logger.info("✅ ML System V4 training completed successfully")
    else:
        logger.error("❌ ML System V4 training failed")
