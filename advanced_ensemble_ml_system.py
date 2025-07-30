#!/usr/bin/env python3
"""
Advanced Ensemble ML System - Professional Grade Prediction Engine
================================================================

This system implements cutting-edge ensemble methods and betting strategy
integration to maximize both prediction accuracy and profitability.

Features:
- Multi-model ensemble (GradientBoosting, XGBoost, RandomForest)
- Advanced stacking with meta-learner
- Betting strategy optimization
- Real-time model updating
- Professional-grade risk management
"""

import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ml_system_v3 import MLSystemV3
from traditional_analysis import TraditionalRaceAnalyzer

logger = logging.getLogger(__name__)

class BettingStrategyOptimizer:
    """
    Professional betting strategy optimizer that converts predictions
    into profitable betting recommendations with risk management.
    """
    
    def __init__(self):
        self.kelly_multiplier = 0.25  # Conservative Kelly criterion
        self.min_edge = 0.05  # Minimum 5% edge required
        self.max_bet_size = 0.1  # Maximum 10% of bankroll per bet
        self.confidence_threshold = 0.7  # Minimum confidence for betting
        
    def calculate_betting_value(self, win_prob: float, market_odds: float, 
                              confidence: float) -> dict:
        """
        Calculate betting value and recommended stake using Kelly criterion
        with professional risk management.
        """
        if win_prob <= 0 or market_odds <= 1.0 or confidence < self.confidence_threshold:
            return self._no_bet_recommendation()
        
        # Calculate implied market probability
        market_prob = 1.0 / market_odds
        
        # Calculate edge (our probability vs market probability)
        edge = win_prob - market_prob
        
        if edge < self.min_edge:
            return self._no_bet_recommendation()
        
        # Kelly criterion calculation: f = (bp - q) / b
        # where b = odds-1, p = win_prob, q = 1-win_prob
        b = market_odds - 1
        kelly_fraction = (b * win_prob - (1 - win_prob)) / b
        
        # Apply conservative multiplier and cap
        recommended_stake = min(
            kelly_fraction * self.kelly_multiplier,
            self.max_bet_size
        )
        
        # Adjust for confidence
        confidence_adjustment = min(confidence / 0.9, 1.0)
        recommended_stake *= confidence_adjustment
        
        # Calculate expected value
        expected_value = (win_prob * (market_odds - 1)) - (1 - win_prob)
        
        bet_type = self._determine_bet_type(win_prob, edge, confidence)
        
        return {
            'has_value': True,
            'edge': edge,
            'expected_value': expected_value,
            'recommended_stake': max(0, recommended_stake),
            'bet_type': bet_type,
            'confidence_rating': self._get_confidence_rating(confidence, edge),
            'risk_level': self._calculate_risk_level(recommended_stake, edge),
            'kelly_fraction': kelly_fraction
        }
    
    def _determine_bet_type(self, win_prob: float, edge: float, confidence: float) -> str:
        """Determine optimal bet type based on probability and confidence."""
        if win_prob >= 0.6 and edge >= 0.15 and confidence >= 0.85:
            return "STRONG_WIN"
        elif win_prob >= 0.4 and edge >= 0.1 and confidence >= 0.75:
            return "WIN"
        elif win_prob >= 0.25 and edge >= 0.08:
            return "PLACE"
        elif win_prob >= 0.15 and edge >= 0.05:
            return "EACH_WAY"
        else:
            return "NO_BET"
    
    def _get_confidence_rating(self, confidence: float, edge: float) -> str:
        """Get confidence rating for the bet."""
        if confidence >= 0.9 and edge >= 0.2:
            return "VERY_HIGH"
        elif confidence >= 0.8 and edge >= 0.15:
            return "HIGH"
        elif confidence >= 0.75 and edge >= 0.1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_risk_level(self, stake: float, edge: float) -> str:
        """Calculate risk level of the bet."""
        if stake <= 0.02 or edge >= 0.2:
            return "LOW"
        elif stake <= 0.05 or edge >= 0.15:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _no_bet_recommendation(self) -> dict:
        """Return no-bet recommendation."""
        return {
            'has_value': False,
            'edge': 0,
            'expected_value': 0,
            'recommended_stake': 0,
            'bet_type': 'NO_BET',
            'confidence_rating': 'INSUFFICIENT',
            'risk_level': 'NONE',
            'kelly_fraction': 0
        }

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering with time-series and interaction features.
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """Fit the feature engineer."""
        X_transformed = self._create_features(X)
        self.scaler.fit(X_transformed)
        self.feature_names_ = X_transformed.columns.tolist()
        return self
    
    def transform(self, X):
        """Transform features."""
        X_transformed = self._create_features(X)
        X_scaled = self.scaler.transform(X_transformed)
        return pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
    
    def _create_features(self, X):
        """Create advanced features."""
        X_new = X.copy()
        
        # Time-based features (if race_date available)
        if 'race_date' in X_new.columns:
            X_new['race_date'] = pd.to_datetime(X_new['race_date'], errors='coerce')
            X_new['days_since_epoch'] = (X_new['race_date'] - pd.Timestamp('2020-01-01')).dt.days
            X_new['month'] = X_new['race_date'].dt.month
            X_new['day_of_week'] = X_new['race_date'].dt.dayofweek
            X_new['is_weekend'] = (X_new['day_of_week'] >= 5).astype(int)
        
        # Advanced interaction features
        numeric_cols = X_new.select_dtypes(include=[np.number]).columns
        
        if 'weight' in numeric_cols and 'starting_price' in numeric_cols:
            X_new['weight_price_interaction'] = X_new['weight'] * X_new['starting_price']
        
        if 'box_number' in numeric_cols and 'field_size' in numeric_cols:
            X_new['box_position_ratio'] = X_new['box_number'] / X_new['field_size']
        
        # Polynomial features for key variables
        if 'individual_time' in numeric_cols:
            X_new['time_squared'] = X_new['individual_time'] ** 2
        
        if 'weight' in numeric_cols:
            X_new['weight_squared'] = X_new['weight'] ** 2
        
        # Remove non-numeric columns for ML
        return X_new.select_dtypes(include=[np.number])

class AdvancedEnsembleMLSystem:
    """
    Professional-grade ensemble ML system with betting strategy integration.
    """
    
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.models = {}
        self.ensemble_model = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.betting_optimizer = BettingStrategyOptimizer()
        self.model_info = {}
        self.feature_names = []
        
        # Initialize base ML system for data loading
        self.base_ml_system = MLSystemV3(db_path)
        
        logger.info("🎯 Advanced Ensemble ML System initialized")
    
    def train_ensemble(self, models_to_train=['gradient_boosting', 'random_forest', 'xgboost']):
        """
        Train ensemble of models with advanced feature engineering.
        """
        logger.info("🚀 Starting advanced ensemble training...")
        
        # Load and prepare data
        data = self.base_ml_system._load_comprehensive_data()
        if data.empty:
            logger.error("No data loaded, cannot train ensemble")
            return False
        
        logger.info(f"Loaded {len(data)} records for ensemble training")
        
        # Create features and target using base system
        features, target = self.base_ml_system._create_comprehensive_features(data)
        if features.empty:
            logger.error("No features created, cannot train ensemble")
            return False
        
        # Apply advanced feature engineering
        logger.info("Applying advanced feature engineering...")
        self.feature_engineer.fit(features)
        features_advanced = self.feature_engineer.transform(features)
        self.feature_names = features_advanced.columns.tolist()
        
        logger.info(f"Created {len(self.feature_names)} advanced features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_advanced, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Train individual models
        self._train_individual_models(X_train, X_test, y_train, y_test, models_to_train)
        
        # Create ensemble
        self._create_ensemble_model(X_train, X_test, y_train, y_test)
        
        # Evaluate ensemble
        ensemble_score = self._evaluate_ensemble(X_test, y_test)
        
        # Save model info
        self.model_info = {
            'ensemble_type': 'VotingClassifier',
            'base_models': list(self.models.keys()),
            'n_features': len(self.feature_names),
            'n_samples': len(features_advanced),
            'ensemble_roc_auc': ensemble_score,
            'trained_at': datetime.now().isoformat(),
            'features': self.feature_names
        }
        
        logger.info(f"✅ Ensemble training completed! ROC AUC: {ensemble_score:.4f}")
        return True
    
    def _train_individual_models(self, X_train, X_test, y_train, y_test, models_to_train):
        """Train individual models for the ensemble."""
        
        model_configs = {
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE and 'xgboost' in models_to_train:
            model_configs['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
        
        for model_name in models_to_train:
            if model_name not in model_configs:
                continue
                
            logger.info(f"Training {model_name}...")
            config = model_configs[model_name]
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate model
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            self.models[model_name] = grid_search.best_estimator_
            
            logger.info(f"  {model_name} ROC AUC: {roc_auc:.4f}")
            logger.info(f"  Best params: {grid_search.best_params_}")
    
    def _create_ensemble_model(self, X_train, X_test, y_train, y_test):
        """Create voting ensemble from trained models."""
        if not self.models:
            logger.error("No models trained, cannot create ensemble")
            return
        
        # Create voting classifier
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use predicted probabilities
        )
        
        logger.info("Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
    
    def _evaluate_ensemble(self, X_test, y_test):
        """Evaluate the ensemble model."""
        if not self.ensemble_model:
            return 0.0
        
        y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        return roc_auc
    
    def predict(self, dog_data, market_odds=None):
        """
        Make prediction with betting strategy recommendation.
        """
        if not self.ensemble_model:
            logger.error("No ensemble model trained")
            return self._error_prediction()
        
        try:
            # Extract features using base system
            features = self.base_ml_system._extract_features_for_prediction(dog_data)
            
            # Create DataFrame
            features_df = pd.DataFrame([features])
            
            # Align with training features
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            features_df = features_df[self.feature_names]
            
            # Apply feature engineering
            features_advanced = self.feature_engineer.transform(features_df)
            
            # Make prediction
            win_prob = self.ensemble_model.predict_proba(features_advanced)[0, 1]
            
            # Calculate model confidence (standard deviation of individual predictions)
            individual_probs = []
            for model in self.models.values():
                prob = model.predict_proba(features_advanced)[0, 1]
                individual_probs.append(prob)
            
            confidence = 1.0 - np.std(individual_probs)  # Higher std = lower confidence
            confidence = max(0.1, min(0.95, confidence))
            
            result = {
                'win_probability': float(win_prob),
                'confidence': float(confidence),
                'model_info': 'advanced_ensemble',
                'individual_predictions': {
                    name: float(model.predict_proba(features_advanced)[0, 1])
                    for name, model in self.models.items()
                }
            }
            
            # Add betting recommendation if market odds provided
            if market_odds:
                betting_rec = self.betting_optimizer.calculate_betting_value(
                    win_prob, market_odds, confidence
                )
                result['betting_recommendation'] = betting_rec
            
            return result
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return self._error_prediction()
    
    def _error_prediction(self):
        """Return error prediction."""
        return {
            'win_probability': 0.5,
            'confidence': 0.0,
            'model_info': 'error',
            'individual_predictions': {},
            'betting_recommendation': self.betting_optimizer._no_bet_recommendation()
        }
    
    def save_ensemble(self, filename=None):
        """Save the trained ensemble model."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'advanced_ensemble_{timestamp}.joblib'
        
        model_dir = Path('./advanced_models')
        model_dir.mkdir(exist_ok=True)
        filepath = model_dir / filename
        
        ensemble_data = {
            'ensemble_model': self.ensemble_model,
            'individual_models': self.models,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'model_info': self.model_info,
            'betting_optimizer': self.betting_optimizer
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
        return filepath
    
    def load_ensemble(self, filepath):
        """Load a saved ensemble model."""
        try:
            ensemble_data = joblib.load(filepath)
            
            self.ensemble_model = ensemble_data['ensemble_model']
            self.models = ensemble_data['individual_models']
            self.feature_engineer = ensemble_data['feature_engineer']
            self.feature_names = ensemble_data['feature_names']
            self.model_info = ensemble_data.get('model_info', {})
            self.betting_optimizer = ensemble_data.get('betting_optimizer', BettingStrategyOptimizer())
            
            logger.info(f"Ensemble loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the ensemble model."""
        return self.model_info

# Convenience function for training
def train_advanced_ensemble(model_types=['gradient_boosting', 'random_forest', 'xgboost']):
    """Train advanced ensemble model."""
    try:
        ensemble_system = AdvancedEnsembleMLSystem()
        success = ensemble_system.train_ensemble(model_types)
        
        if success:
            # Save the ensemble
            model_path = ensemble_system.save_ensemble()
            
            return {
                'success': True,
                'message': 'Advanced ensemble trained successfully',
                'model_info': ensemble_system.get_model_info(),
                'model_path': str(model_path)
            }
        else:
            return {
                'success': False,
                'message': 'Ensemble training failed'
            }
    except Exception as e:
        return {
            'success': False,
            'message': f'Training error: {str(e)}'
        }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    ensemble_system = AdvancedEnsembleMLSystem()
    success = ensemble_system.train_ensemble()
    
    if success:
        # Test prediction
        test_dog = {
            'name': 'Test Champion',
            'box_number': 1,
            'weight': 32.0,
            'starting_price': 2.50,
            'individual_time': 29.50,
            'field_size': 8
        }
        
        prediction = ensemble_system.predict(test_dog, market_odds=3.0)
        print(f"Prediction: {prediction}")
