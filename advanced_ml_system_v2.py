#!/usr/bin/env python3
"""
Advanced ML System v2.0
========================

Improved machine learning architecture specifically designed for greyhound racing
prediction with dynamic ensemble weighting and performance-based model selection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib
import json
from datetime import datetime, timedelta
import sqlite3
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class AdvancedMLSystemV2:
    def __init__(self, db_path="greyhound_racing_data.db", skip_auto_load=False):
        self.db_path = db_path
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.scaler = RobustScaler()
        self._models_freshly_trained = False
        
        # Auto-load trained models if available (unless skipped)
        if not skip_auto_load:
            self._auto_load_models()
        
        # Advanced model configurations
        self.model_configs = {
            'random_forest_optimized': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': 42,
                    'class_weight': 'balanced'  # Handle class imbalance
                }
            },
            'gradient_boosting_optimized': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_samples_split': 4,
                    'min_samples_leaf': 1,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (100, 50, 25),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 1000,
                    'random_state': 42,
                    'early_stopping': True
                }
            },
            'logistic_regression_optimized': {
                'model': LogisticRegression,
                'params': {
                    'C': 0.1,
                    'penalty': 'l2',
                    'solver': 'liblinear',
                    'max_iter': 2000,
                    'random_state': 42,
                    'class_weight': 'balanced'
                }
            }
        }
        
    def prepare_training_data(self, enhanced_features_list):
        """Prepare training data with enhanced features"""
        # Convert enhanced features to DataFrame
        training_data = []
        
        for feature_dict in enhanced_features_list:
            if 'target' in feature_dict and 'features' in feature_dict:
                row = feature_dict['features'].copy()
                row['target'] = feature_dict['target']
                training_data.append(row)
        
        if not training_data:
            return None, None, None, None
        
        df = pd.DataFrame(training_data)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'target']
        X = df[feature_columns]
        y = df['target']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y, feature_columns, df
    
    def train_advanced_models(self, X, y, feature_columns):
        """Train advanced models with cross-validation"""
        print("🚀 Training advanced ML models...")
        
        # Time series split for temporal validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Validate missing features
        missing_features = [col for col in feature_columns if col not in X.columns]
        if missing_features:
            print(f"⚠️ Warning: Missing features detected - {missing_features}")

        # Check for correct feature data types and ranges
        out_of_range_features = []
        for col in feature_columns:
            if col in X.columns:
                if (X[col] < 0).any() or (X[col] > 1).any():  # Assuming normalized or constrained [0,1] range
                    out_of_range_features.append(col)
        if out_of_range_features:
            print(f"⚠️ Warning: Out-of-range values detected in features - {out_of_range_features}")

        results = {}
        
        # Fit scaler on all training data once for consistency
        print("   Fitting scaler on training data...")
        self.scaler.fit(X)
        
        # Print scaler statistics for validation
        print("   Scaler statistics:")
        for i, col in enumerate(feature_columns):
            center = self.scaler.center_[i] if hasattr(self.scaler, 'center_') else 'N/A'
            scale = self.scaler.scale_[i] if hasattr(self.scaler, 'scale_') else 'N/A'
            print(f"     {col}: center={center:.4f}, scale={scale:.4f}")
        
        for model_name, config in self.model_configs.items():
            print(f"   Training {model_name}...")
            
            model_class = config['model']
            params = config['params']
            
            # Create model
            model = model_class(**params)
            
            # Apply consistent scaling strategy
            # Scale features for models that benefit from scaling
            if model_name in ['neural_network', 'logistic_regression_optimized', 'gradient_boosting_optimized']:
                X_train = self.scaler.transform(X)
                use_scaling = True
            else:
                X_train = X.values  # Convert to numpy array for consistency
                use_scaling = False
            
            # Cross-validation scores with proper scaling
            if use_scaling:
                # For scaled models, we need to scale in each CV fold
                from sklearn.pipeline import Pipeline
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', model)
                ])
                cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
                cv_auc_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y, cv=tscv, scoring='accuracy')
                cv_auc_scores = cross_val_score(model, X_train, y, cv=tscv, scoring='roc_auc')
            
            # Train final model on all data
            model.fit(X_train, y)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = list(zip(feature_columns, model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
            elif hasattr(model, 'coef_'):
                feature_importance = list(zip(feature_columns, np.abs(model.coef_[0])))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
            else:
                feature_importance = None
            
            results[model_name] = {
                'model': model,
                'use_scaling': use_scaling,
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'cv_auc_mean': cv_auc_scores.mean(),
                'cv_auc_std': cv_auc_scores.std(),
                'feature_importance': feature_importance
            }
            
            print(f"     ✅ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"     ✅ CV AUC: {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")

            # Store detailed cross-validation results for further analysis
            print(f"     📈 Cross-validation detailed results: {cv_scores}")
            print(f"     📊 AUC scores: {cv_auc_scores}")
        
        # Calculate dynamic weights based on performance
        self.calculate_dynamic_weights(results)
        
        self.models = {name: result['model'] for name, result in results.items()}
        
        # Store scaling information for each model
        self._model_scaling = {name: result['use_scaling'] for name, result in results.items()}
        
        # Store feature columns used in training
        self._model_feature_columns = feature_columns
        
        # Mark that we have fresh trained models (not from registry)
        self._models_freshly_trained = True
        
        print(f"\n🎉 Training completed! Ensemble contains {len(self.models)} models.")
        print(f"📊 Feature columns: {len(feature_columns)}")
        print(f"🔄 Models override any previously loaded registry models.")
        
        return results
    
    def calculate_dynamic_weights(self, results):
        """Calculate dynamic weights based on model performance"""
        # Weight models based on CV AUC scores
        auc_scores = {name: result['cv_auc_mean'] for name, result in results.items()}
        
        # Convert to weights (higher AUC = higher weight)
        total_auc = sum(auc_scores.values())
        base_weights = {name: score / total_auc for name, score in auc_scores.items()}
        
        # Apply performance bonuses/penalties
        adjusted_weights = {}
        for name, weight in base_weights.items():
            result = results[name]
            
            # Bonus for high accuracy and low variance
            accuracy_bonus = max(0, (result['cv_accuracy_mean'] - 0.6) * 0.5)
            stability_bonus = max(0, (0.1 - result['cv_accuracy_std']) * 0.3)
            
            adjusted_weight = weight + accuracy_bonus + stability_bonus
            adjusted_weights[name] = adjusted_weight
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        self.model_weights = {name: weight / total_weight for name, weight in adjusted_weights.items()}
        
        print("\n🎯 Dynamic Model Weights:")
        for name, weight in self.model_weights.items():
            print(f"   {name}: {weight:.3f}")
    
    def predict_with_ensemble(self, features_dict):
        """Make predictions using weighted ensemble with enhanced feature handling"""
        if not self.models:
            return 0.5  # Default probability
        
        # No feature mapping needed since input features match model expectations
        mapped_features = features_dict
        
        # Get expected feature list
        base_features = [
            'weighted_recent_form', 'speed_trend', 'speed_consistency', 'venue_win_rate',
            'venue_avg_position', 'venue_experience', 'distance_win_rate', 'distance_avg_time',
            'box_position_win_rate', 'box_position_avg', 'recent_momentum', 'competitive_level',
            'position_consistency', 'top_3_rate', 'break_quality'
        ]
        
        # Debug feature matching
        if not hasattr(self, '_debug_shown'):
            print(f"🔍 ML Model feature validation:")
            if hasattr(self, '_model_feature_columns'):
                print(f"   Model-specific features: {self._model_feature_columns}")
            print(f"   Base features: {base_features}")
            print(f"   Input features: {sorted(list(features_dict.keys()))}")
            self._debug_shown = True
            
        # Get expected features and verify
        expected_features = self._model_feature_columns if hasattr(self, '_model_feature_columns') else base_features
        if not hasattr(self, '_features_verified'):
            missing = [f for f in expected_features if f not in mapped_features]
            if missing:
                print(f"⚠️ Missing expected features: {missing}")
                print("   This may impact prediction quality")
            self._features_verified = True

        # Create feature vector with EXACT ordering and naming the model expects
        feature_values = {}
        missing_features = []

        for feature_name in expected_features:
            if feature_name in mapped_features:
                value = mapped_features[feature_name]
                feature_values[feature_name] = float(value)  # Ensure numeric
            else:
                value = 0.0  # Default for missing features
                feature_values[feature_name] = value
                missing_features.append(feature_name)

        if missing_features and not hasattr(self, '_missing_warned'):
            print(f"⚠️ Missing features (defaulting to 0.0): {missing_features}")
            self._missing_warned = True

        if not feature_values:
            return 0.5  # No valid features

        # Create DataFrame with EXACT ordering of features
        import pandas as pd
        feature_df = pd.DataFrame([feature_values], columns=expected_features)
        feature_df = feature_df[expected_features]  # Ensure exact column order
        
        # Debug output for feature values
        print("\n   🔎 Feature values:")
        for col in expected_features:
            print(f"     {col}: {feature_df[col].iloc[0]:.4f}")
        
        # Suppress debug output to reduce noise
        # print(f"📊 Feeding {len(feature_values)} features to ML model: {feature_values[:5]}...")  # Show first 5 values for debugging
        
        weighted_predictions = []
        
        for model_name, model in self.models.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                
                # Check if this model uses scaling
                use_scaling = hasattr(self, '_model_scaling') and self._model_scaling.get(model_name, False)
                
                if use_scaling:
                    try:
                        # Scale features and preserve column names
                        features_scaled = self.scaler.transform(feature_df)
                        features_scaled_df = pd.DataFrame(
                            features_scaled,
                            columns=expected_features
                        )
                        pred_proba = model.predict_proba(features_scaled_df)[0, 1]
                        print(f"   🎯 {model_name}: {pred_proba:.4f} (weight: {weight:.2f}, scaled)")
                    except Exception as e:
                        print(f"⚠️ Error predicting with {model_name} (scaled): {e}")
                        pred_proba = 0.5
                else:
                    try:
                        # Use raw features
                        pred_proba = model.predict_proba(feature_df)[0, 1]
                        print(f"   🎯 {model_name}: {pred_proba:.4f} (weight: {weight:.2f}, raw)")
                    except Exception as e:
                        print(f"⚠️ Error predicting with {model_name}: {e}")
                        pred_proba = 0.5
                
                print(f"   🎯 {model_name}: {pred_proba:.4f} (weight: {weight:.2f})")
                weighted_predictions.append(pred_proba * weight)
        
        # Return weighted average
        if weighted_predictions:
            final_prediction = sum(weighted_predictions)
            # Apply confidence adjustment based on data quality
            data_quality = features_dict.get('data_quality', 0.5)
            adjusted_prediction = final_prediction * (0.7 + 0.3 * data_quality)
            
            print(f"   📊 Final prediction: {final_prediction:.4f} (adjusted: {adjusted_prediction:.4f})")
            return min(0.95, max(0.05, adjusted_prediction))  # Constrain to reasonable range
        
        return 0.5
    
    def _enhance_feature_vector(self, features_dict, feature_array):
        """Enhance feature vector to improve model differentiation"""
        enhanced_array = feature_array.copy()
        
        # Add interaction features for better differentiation
        box_number = features_dict.get('box_number', 0)
        form_score = features_dict.get('weighted_recent_form', 5.0)
        venue_rate = features_dict.get('venue_win_rate', 0.0)
        data_quality = features_dict.get('data_quality', 0.5)
        
        # Create interaction features
        box_form_interaction = (9 - box_number) * (6 - form_score) / 25  # Normalize to 0-1 range
        box_venue_interaction = (9 - box_number) * venue_rate
        quality_form_interaction = data_quality * (6 - form_score) / 6
        
        # Add synthetic features to increase differentiation
        synthetic_features = np.array([
            box_form_interaction,
            box_venue_interaction, 
            quality_form_interaction,
            box_number / 8.0,  # Normalized box position
            (6 - form_score) / 6.0 if form_score <= 6 else 0,  # Normalized form (inverted)
        ]).reshape(1, -1)
        
        # Concatenate original and synthetic features
        enhanced_array = np.concatenate([enhanced_array, synthetic_features], axis=1)
        
        return enhanced_array
    
    def update_performance_tracking(self, predictions, actual_results):
        """Update performance tracking for continuous learning"""
        timestamp = datetime.now().isoformat()
        
        # Calculate accuracy for this batch
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred, actual in zip(predictions, actual_results):
            if (pred > 0.5 and actual == 1) or (pred <= 0.5 and actual == 0):
                correct_predictions += 1
        
        batch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Store performance data
        performance_entry = {
            'timestamp': timestamp,
            'batch_accuracy': batch_accuracy,
            'sample_size': total_predictions,
            'model_weights': self.model_weights.copy()
        }
        
        if 'recent_performance' not in self.performance_history:
            self.performance_history['recent_performance'] = []
        
        self.performance_history['recent_performance'].append(performance_entry)
        
        # Keep only last 50 entries
        self.performance_history['recent_performance'] = self.performance_history['recent_performance'][-50:]
        
        # Check if retraining is needed
        if len(self.performance_history['recent_performance']) >= 10:
            recent_accuracies = [entry['batch_accuracy'] for entry in self.performance_history['recent_performance'][-10:]]
            avg_recent_accuracy = np.mean(recent_accuracies)
            
            if avg_recent_accuracy < 0.55:  # Performance degradation threshold
                print(f"⚠️ Performance degradation detected: {avg_recent_accuracy:.3f}")
                print("   Consider retraining models with recent data")
        
        return batch_accuracy
    
    def save_models(self, filepath_prefix="advanced_ml_model"):
        """Save trained models and metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'scaler': self.scaler,
            'performance_history': self.performance_history,
            'timestamp': timestamp
        }
        
        filepath = f"{filepath_prefix}_{timestamp}.joblib"
        joblib.dump(model_data, filepath)
        
        print(f"✅ Models saved to {filepath}")
        return filepath
    
    def load_models(self, filepath):
        """Load trained models and metadata"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.model_weights = model_data['model_weights']
            self.scaler = model_data['scaler']
            self.performance_history = model_data.get('performance_history', {})
            
            print(f"✅ Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def generate_prediction_confidence(self, features_dict):
        """Generate confidence score for prediction"""
        if not self.models:
            return 0.3  # Low confidence if no models
        
        # Get predictions from all models
        predictions = []
        feature_array = np.array(list(features_dict.values())).reshape(1, -1)
        feature_array = np.nan_to_num(feature_array, nan=np.median(feature_array))
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['neural_network', 'logistic_regression_optimized']:
                    features_scaled = self.scaler.transform(feature_array)
                    pred_proba = model.predict_proba(features_scaled)[0, 1]
                else:
                    pred_proba = model.predict_proba(feature_array)[0, 1]
                
                predictions.append(pred_proba)
            except:
                predictions.append(0.5)
        
        if not predictions:
            return 0.3
        
        # Calculate confidence based on prediction variance
        prediction_std = np.std(predictions)
        data_quality = features_dict.get('data_quality', 0.5)
        
        # Lower variance = higher confidence
        base_confidence = max(0.3, 1.0 - (prediction_std * 2))
        
        # Adjust for data quality
        final_confidence = base_confidence * (0.6 + 0.4 * data_quality)
        
        return min(0.95, max(0.3, final_confidence))
    
    def _map_enhanced_to_comprehensive_features(self, enhanced_features):
        """Map Enhanced Pipeline V2 features to comprehensive model features"""
        mapped = {}
        
        # Direct mappings
        if 'box_number' in enhanced_features:
            mapped['current_box'] = enhanced_features['box_number']
            
        # Statistical features
        if 'weighted_recent_form' in enhanced_features:
            # Lower form score is better (1-6 scale)
            form_val = enhanced_features['weighted_recent_form']
            if form_val <= 6:  # Valid form score
                mapped['recent_form_avg'] = form_val
                mapped['avg_position'] = form_val  # Approximate
                mapped['long_term_form_trend'] = max(0, 3 - form_val)  # Convert to trend
            else:
                mapped['recent_form_avg'] = 5.0
                mapped['avg_position'] = 5.0
                mapped['long_term_form_trend'] = 0
        
        if 'venue_win_rate' in enhanced_features:
            mapped['win_rate'] = enhanced_features['venue_win_rate']
            mapped['place_rate'] = enhanced_features['venue_win_rate'] * 2.5  # Estimate
        
        if 'venue_experience' in enhanced_features:
            mapped['venue_experience'] = enhanced_features['venue_experience']
            mapped['grade_experience'] = enhanced_features['venue_experience']  # Approximate
        
        if 'speed_consistency' in enhanced_features:
            mapped['time_consistency'] = enhanced_features['speed_consistency']
            mapped['position_consistency'] = enhanced_features['speed_consistency']
        
        if 'speed_trend' in enhanced_features:
            mapped['time_improvement_trend'] = enhanced_features['speed_trend']
        
        if 'distance_avg_time' in enhanced_features:
            mapped['avg_time'] = enhanced_features['distance_avg_time']
            mapped['best_time'] = enhanced_features['distance_avg_time'] * 0.95  # Estimate
        
        if 'distance_win_rate' in enhanced_features:
            mapped['distance_specialization'] = enhanced_features['distance_win_rate']
        
        if 'box_position_win_rate' in enhanced_features:
            mapped['box_win_rate'] = enhanced_features['box_position_win_rate']
        
        # Enhanced mapping for traditional score features
        form_score = enhanced_features.get('weighted_recent_form', 5.0)
        data_quality = enhanced_features.get('data_quality', 0.5)
        
        # Traditional scores (scale 0-1, higher is better)
        base_score = max(0, 1 - (form_score / 6))  # Convert 1-6 form to 0-1 score
        mapped['traditional_overall_score'] = base_score * data_quality
        mapped['traditional_performance_score'] = base_score * 0.9
        mapped['traditional_form_score'] = base_score
        mapped['traditional_consistency_score'] = enhanced_features.get('speed_consistency', 0.5)
        mapped['traditional_confidence_level'] = data_quality
        mapped['traditional_class_score'] = enhanced_features.get('competitive_level', 0.5) / 10
        mapped['traditional_fitness_score'] = max(0, 1 - abs(enhanced_features.get('speed_trend', 0)))
        mapped['traditional_experience_score'] = min(1, enhanced_features.get('venue_experience', 0) / 10)
        mapped['traditional_trainer_score'] = enhanced_features.get('trainer_impact', 0.5)
        mapped['traditional_track_condition_score'] = 0.7  # Default
        mapped['traditional_distance_score'] = enhanced_features.get('distance_win_rate', 0.5)
        
        # Weight features (defaults)
        mapped['current_weight'] = 30.0  # Typical weight
        mapped['avg_weight'] = 30.0
        mapped['weight_consistency'] = 0.8
        mapped['weight_vs_avg'] = 0.0
        
        # Weather features (defaults)
        mapped['temperature'] = 20.0
        mapped['humidity'] = 60.0
        mapped['wind_speed'] = 5.0
        mapped['pressure'] = 1013.25
        mapped['weather_adjustment_factor'] = 1.0
        
        # Weather categorical features (defaults to optimal conditions)
        mapped['weather_clear'] = 1
        mapped['weather_cloudy'] = 0
        mapped['weather_rain'] = 0
        mapped['weather_fog'] = 0
        mapped['temp_optimal'] = 1
        mapped['temp_cold'] = 0
        mapped['temp_cool'] = 0
        mapped['temp_warm'] = 0
        mapped['temp_hot'] = 0
        mapped['wind_light'] = 1
        mapped['wind_calm'] = 0
        mapped['wind_moderate'] = 0
        mapped['wind_strong'] = 0
        mapped['humidity_normal'] = 1
        mapped['humidity_low'] = 0
        mapped['humidity_high'] = 0
        
        # Experience and performance features
        mapped['weather_experience_count'] = 5
        mapped['weather_performance'] = 0.6
        mapped['days_since_last'] = 14  # Default
        mapped['competition_strength'] = enhanced_features.get('competitive_level', 5.0)
        mapped['field_size'] = 8  # Typical field size
        mapped['historical_races_count'] = max(1, enhanced_features.get('venue_experience', 1))
        
        # Encoded features (defaults)
        mapped['venue_encoded'] = 1
        mapped['track_condition_encoded'] = 1
        mapped['grade_encoded'] = 5
        mapped['distance_numeric'] = 500  # Typical distance
        
        # Fitness and key factors
        mapped['fitness_score'] = max(0, 1 - abs(enhanced_features.get('speed_trend', 0)))
        mapped['traditional_key_factors_count'] = 3
        mapped['traditional_risk_factors_count'] = 1
        
        # Market confidence (convert from form)
        if 'weighted_recent_form' in enhanced_features:
            form_val = enhanced_features['weighted_recent_form']
            mapped['market_confidence'] = max(0.1, 1 - (form_val / 6))  # Better form = higher confidence
        else:
            mapped['market_confidence'] = 0.5
        
        # Current odds (estimated from form and other factors)
        market_conf = mapped.get('market_confidence', 0.5)
        mapped['current_odds_log'] = np.log(max(1.5, 10 - (market_conf * 8)))
        
        return mapped
    
    def _auto_load_models(self):
        """Automatically load the best trained model from the model registry"""
        try:
            from model_registry import get_model_registry
            
            # Get the global model registry
            registry = get_model_registry()
            
            # Try to get the best model from registry
            best_model_result = registry.get_best_model()
            if best_model_result:
                model, scaler, metadata = best_model_result
                
                print(f"📂 Auto-loading best model from registry: {metadata.model_id}")
                print(f"   📊 Performance: Acc={metadata.accuracy:.3f}, AUC={metadata.auc:.3f}, F1={metadata.f1_score:.3f}")
                print(f"   🔄 Training: {metadata.training_timestamp[:10]} ({metadata.training_samples:,} samples)")
                
                # Set up the model in ensemble format
                model_name = metadata.model_name or 'registry_model'
                self.models = {model_name: model}
                self.model_weights = {model_name: 1.0}
                self.scaler = scaler
                
                # Store expected feature columns for proper mapping
                if metadata.feature_names:
                    self._model_feature_columns = metadata.feature_names
                
                # Store metadata for reference
                self._current_model_metadata = metadata
                
                print(f"✅ Registry model loaded successfully: {model_name}")
                return
            
            print("⚠️ No models found in registry, falling back to legacy loading...")
            
        except ImportError:
            print("⚠️ Model registry not available, using legacy model loading...")
        except Exception as e:
            print(f"⚠️ Error loading from model registry: {e}")
            print("   Falling back to legacy model loading...")
        
        # Fallback to legacy loading
        self._legacy_auto_load_models()
    
    def _legacy_auto_load_models(self):
        """Legacy model loading method as fallback"""
        import os
        import glob
        
        print("🔍 Searching for trained models...")
        
        # Keep track of all candidate models
        candidate_models = []
        
        # Check comprehensive trained models directory
        model_dir = "comprehensive_trained_models"
        if os.path.exists(model_dir):
            model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
            for model_file in model_files:
                try:
                    model_data = joblib.load(model_file)
                    if not isinstance(model_data, dict):
                        print(f"⚠️ Skipping {model_file}: Invalid format (not a dictionary)")
                        continue
                    
                    # Validate comprehensive model format
                    if 'model' in model_data:
                        required_keys = ['model', 'feature_columns']
                        if not all(key in model_data for key in required_keys):
                            print(f"⚠️ Skipping {model_file}: Missing required keys {required_keys}")
                            continue
                        
                        candidate_models.append({
                            'file': model_file,
                            'type': 'comprehensive',
                            'data': model_data,
                            'features': len(model_data['feature_columns']),
                            'timestamp': os.path.getctime(model_file)
                        })
                    # Validate advanced ensemble format
                    elif 'models' in model_data and 'model_weights' in model_data:
                        if not model_data['models'] or not isinstance(model_data['models'], dict):
                            print(f"⚠️ Skipping {model_file}: Invalid or empty models")
                            continue
                        
                        candidate_models.append({
                            'file': model_file,
                            'type': 'ensemble',
                            'data': model_data,
                            'features': len(model_data['models']),  # Number of ensemble models
                            'timestamp': os.path.getctime(model_file)
                        })
                except Exception as e:
                    print(f"⚠️ Could not load {model_file}: {e}")
                    continue
        
        # Check for advanced ML system models in root directory
        advanced_model_files = glob.glob("advanced_ml_model_*.joblib")
        for model_file in advanced_model_files:
            try:
                model_data = joblib.load(model_file)
                if not isinstance(model_data, dict) or 'models' not in model_data:
                    continue
                
                candidate_models.append({
                    'file': model_file,
                    'type': 'ensemble',
                    'data': model_data,
                    'features': len(model_data['models']),
                    'timestamp': os.path.getctime(model_file)
                })
            except Exception as e:
                print(f"⚠️ Could not load {model_file}: {e}")
                continue
        
        if not candidate_models:
            print("⚠️ No valid models found - will use heuristic predictions")
            return
        
        # First try models with most features
        candidate_models.sort(key=lambda x: (-x['features'], -x['timestamp']))  # Most features, then most recent
        
        for candidate in candidate_models:
            try:
                model_file = candidate['file']
                model_data = candidate['data']
                model_type = candidate['type']
                
                print(f"📂 Attempting to load: {os.path.basename(model_file)}")
                print(f"   Type: {model_type}, Features: {candidate['features']}")
                
                if model_type == 'comprehensive':
                    # Convert comprehensive model to ensemble format
                    model_name = model_data.get('model_name', 'comprehensive_model')
                    self.models = {model_name: model_data['model']}
                    self.model_weights = {model_name: 1.0}
                    self.scaler = model_data.get('scaler', self.scaler)
                    self._model_feature_columns = model_data['feature_columns']
                    
                    accuracy = model_data.get('accuracy', 'N/A')
                    auc = model_data.get('auc', 'N/A')
                    print(f"✅ Legacy model loaded: {model_name}")
                    print(f"   📈 Accuracy: {accuracy}, AUC: {auc}")
                    return
                    
                elif model_type == 'ensemble':
                    # Load ensemble model components
                    self.models = model_data['models']
                    self.model_weights = model_data.get('model_weights', {})
                    self.scaler = model_data.get('scaler', self.scaler)
                    self.performance_history = model_data.get('performance_history', {})
                    
                    # Ensure weights exist for all models
                    if not self.model_weights or set(self.model_weights.keys()) != set(self.models.keys()):
                        print("   ⚠️ Weights missing or mismatched - using equal weights")
                        weight = 1.0 / len(self.models)
                        self.model_weights = {name: weight for name in self.models.keys()}
                    
                    print(f"✅ Ensemble model loaded with {len(self.models)} sub-models")
                    print(f"   📊 Models: {list(self.models.keys())}")
                    print(f"   ⚖️ Weights: {', '.join(f'{k}: {v:.2f}' for k, v in self.model_weights.items())}")
                    return
                    
            except Exception as e:
                print(f"⚠️ Failed to load {candidate['file']}: {e}")
                continue
        
        print("⚠️ All model loading attempts failed - using heuristic predictions")

def main():
    """Demonstration of advanced ML system"""
    ml_system = AdvancedMLSystemV2()
    
    print("🤖 Advanced ML System v2.0 Demonstration")
    print("=" * 50)
    
    # This would typically be called with real enhanced features
    print("📊 System initialized with optimized model configurations")
    print("✅ Ready for training with enhanced features")
    
    # Show model configurations
    print("\n🛠️ Model Configurations:")
    for model_name, config in ml_system.model_configs.items():
        print(f"   {model_name}: {config['model'].__name__}")

if __name__ == "__main__":
    main()
