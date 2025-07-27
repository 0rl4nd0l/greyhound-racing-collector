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
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.scaler = RobustScaler()
        
        # Auto-load trained models if available
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
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            print(f"   Training {model_name}...")
            
            model_class = config['model']
            params = config['params']
            
            # Create model
            model = model_class(**params)
            
            # Scale features for neural network and logistic regression
            if model_name in ['neural_network', 'logistic_regression_optimized']:
                X_scaled = self.scaler.fit_transform(X)
                X_train = X_scaled
            else:
                X_train = X
            
            # Cross-validation scores
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
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'cv_auc_mean': cv_auc_scores.mean(),
                'cv_auc_std': cv_auc_scores.std(),
                'feature_importance': feature_importance
            }
            
            print(f"     ✅ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"     ✅ CV AUC: {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
        
        # Calculate dynamic weights based on performance
        self.calculate_dynamic_weights(results)
        
        self.models = {name: result['model'] for name, result in results.items()}
        
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
        
        # Get the first model to determine expected feature names and order
        first_model = list(self.models.values())[0]
        
        # Try to get feature names from model if available
        if hasattr(first_model, 'feature_names_in_'):
            expected_features = list(first_model.feature_names_in_)
            # Only show debug info once per session
            if not hasattr(self, '_debug_shown'):
                print(f"🔍 ML Model expects {len(expected_features)} features: {expected_features}")
                self._debug_shown = True
        else:
            # Fallback to a reasonable feature order
            expected_features = [
                'weighted_recent_form', 'speed_trend', 'speed_consistency', 'venue_win_rate',
                'venue_avg_position', 'venue_experience', 'distance_win_rate', 'distance_avg_time',
                'box_position_win_rate', 'box_position_avg', 'recent_momentum', 'competitive_level',
                'position_consistency', 'top_3_rate', 'break_quality'
            ]
            if not hasattr(self, '_debug_shown'):
                print(f"🔍 Using fallback features: {len(expected_features)} features")
                self._debug_shown = True
        
        # Create feature vector with EXACT ordering and naming the model expects
        feature_values = []
        missing_features = []
        
        for feature_name in expected_features:
            if feature_name in features_dict:
                value = features_dict[feature_name]
                feature_values.append(float(value))  # Ensure numeric
            else:
                feature_values.append(0.0)  # Default value for missing features
                missing_features.append(feature_name)
        
        if missing_features:
            print(f"⚠️ Missing features (defaulting to 0.0): {missing_features}")
        
        if not feature_values:
            return 0.5  # No valid features
        
        # Create DataFrame with EXACT feature names and order the model expects
        import pandas as pd
        feature_df = pd.DataFrame([feature_values], columns=expected_features)
        
        # Ensure feature names are properly set for sklearn compatibility
        feature_df.columns = expected_features
        
        # Suppress debug output to reduce noise
        # print(f"📊 Feeding {len(feature_values)} features to ML model: {feature_values[:5]}...")  # Show first 5 values for debugging
        
        weighted_predictions = []
        
        for model_name, model in self.models.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                
                # Scale features for neural network and logistic regression
                if model_name in ['neural_network', 'logistic_regression_optimized']:
                    try:
                        # Use DataFrame directly to preserve feature names
                        features_scaled_df = pd.DataFrame(
                            self.scaler.transform(feature_df), 
                            columns=expected_features
                        )
                        pred_proba = model.predict_proba(features_scaled_df)[0, 1]
                    except Exception as e:
                        pred_proba = 0.5
                else:
                    try:
                        # Use DataFrame directly to preserve feature names
                        pred_proba = model.predict_proba(feature_df)[0, 1]
                    except Exception as e:
                        pred_proba = 0.5
                
                weighted_predictions.append(pred_proba * weight)
        
        # Return weighted average
        if weighted_predictions:
            final_prediction = sum(weighted_predictions)
            # Apply confidence adjustment based on data quality
            data_quality = features_dict.get('data_quality', 0.5)
            adjusted_prediction = final_prediction * (0.7 + 0.3 * data_quality)
            
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
    
    def _auto_load_models(self):
        """Automatically load the most recent trained models if available"""
        import os
        import glob
        
        # Check for comprehensive trained models
        model_dir = "comprehensive_trained_models"
        if os.path.exists(model_dir):
            model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
            if model_files:
                # Get the most recent model
                latest_model = max(model_files, key=os.path.getctime)
                
                try:
                    model_data = joblib.load(latest_model)
                    
                    # Convert single comprehensive model to ensemble format
                    if isinstance(model_data, dict) and 'model' in model_data:
                        print(f"📂 Auto-loading trained model: {os.path.basename(latest_model)}")
                        
                        # Create ensemble from single model by using it as the primary model
                        model_name = model_data.get('model_name', 'comprehensive_model')
                        self.models = {model_name: model_data['model']}
                        
                        # Set equal weights (single model gets full weight)
                        self.model_weights = {model_name: 1.0}
                        
                        # Load scaler if available
                        if 'scaler' in model_data:
                            self.scaler = model_data['scaler']
                        
                        print(f"✅ Loaded model: {model_name} (accuracy: {model_data.get('accuracy', 'N/A'):.3f})")
                        return
                        
                except Exception as e:
                    print(f"⚠️ Warning: Could not load model {latest_model}: {e}")
        
        # Check for advanced ML system models
        advanced_model_files = glob.glob("advanced_ml_model_*.joblib")
        if advanced_model_files:
            latest_advanced = max(advanced_model_files, key=os.path.getctime)
            try:
                self.load_models(latest_advanced)
                return
            except Exception as e:
                print(f"⚠️ Warning: Could not load advanced model {latest_advanced}: {e}")
        
        print("🔍 No pre-trained models found - using heuristic predictions")

def main():
    """Demonstration of advanced ML system"""
    ml_system = AdvancedMLSystem()
    
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
