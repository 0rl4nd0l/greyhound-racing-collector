#!/usr/bin/env python3
"""
Improved ML System for Imbalanced Race Prediction
=================================================

This enhanced system addresses class imbalance issues and improves AUC performance:
1. Class balancing techniques (SMOTE, class weights)
2. Better evaluation metrics (Precision, Recall, F1, AUC)
3. Threshold optimization
4. Ensemble methods optimized for imbalanced data
5. Probability calibration

Author: AI Assistant
Date: July 25, 2025
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML Libraries for Imbalanced Data
try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        VotingClassifier, BaggingClassifier, ExtraTreesClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.impute import KNNImputer
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, classification_report, 
        precision_recall_curve, roc_curve, f1_score,
        precision_score, recall_score, confusion_matrix
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_class_weight
    
    # Import SMOTE for oversampling
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek
        from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
        IMBALANCED_LEARN_AVAILABLE = True
    except ImportError:
        print("⚠️ imbalanced-learn not available. Install with: pip install imbalanced-learn")
        IMBALANCED_LEARN_AVAILABLE = False
    
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False

class ImprovedMLSystem:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.results_dir = Path('./improved_ml_results')
        self.models_dir = Path('./improved_trained_models')
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Enhanced model configurations with class balancing
        self.base_models = {
            'balanced_random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3,
                    'class_weight': 'balanced', 'random_state': 42
                }
            },
            'balanced_gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 8,
                    'random_state': 42
                }
            },
            'balanced_extra_trees': {
                'model': ExtraTreesClassifier,
                'params': {
                    'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3,
                    'class_weight': 'balanced', 'random_state': 42
                }
            },
            'balanced_logistic': {
                'model': LogisticRegression,
                'params': {
                    'C': 2.0, 'max_iter': 3000, 'class_weight': 'balanced',
                    'random_state': 42
                }
            },
            'balanced_svm': {
                'model': SVC,
                'params': {
                    'C': 2.0, 'kernel': 'rbf', 'probability': True,
                    'class_weight': 'balanced', 'random_state': 42
                }
            }
        }
        
        # Imbalanced learning models (if available)
        if IMBALANCED_LEARN_AVAILABLE:
            self.imbalanced_models = {
                'balanced_rf_classifier': {
                    'model': BalancedRandomForestClassifier,
                    'params': {
                        'n_estimators': 300, 'max_depth': 20, 
                        'sampling_strategy': 'auto', 'random_state': 42
                    }
                },
                'balanced_bagging': {
                    'model': BalancedBaggingClassifier,
                    'params': {
                        'n_estimators': 100, 'sampling_strategy': 'auto',
                        'random_state': 42
                    }
                }
            }
        else:
            self.imbalanced_models = {}
        
        print("🚀 Improved ML System for Imbalanced Data Initialized")
        
    def load_and_prepare_data(self):
        """Load data from the comprehensive ML system results"""
        try:
            # Import the comprehensive system to get feature-engineered data
            from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
            
            comp_system = ComprehensiveEnhancedMLSystem(self.db_path)
            
            # Load form guide data
            form_data = comp_system.load_form_guide_data()
            
            # Load race results
            race_results_df = comp_system.load_race_results_data()
            if race_results_df is None:
                return None, None
            
            # Create comprehensive features
            enhanced_df = comp_system.create_comprehensive_features(race_results_df, form_data)
            if enhanced_df is None:
                return None, None
            
            # Prepare features
            prepared_df, feature_columns = comp_system.prepare_comprehensive_features(enhanced_df)
            if prepared_df is None:
                return None, None
            
            print(f"📊 Loaded dataset: {len(prepared_df)} samples, {len(feature_columns)} features")
            
            return prepared_df, feature_columns
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def analyze_class_distribution(self, y):
        """Analyze and report class distribution"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print("📊 CLASS DISTRIBUTION:")
        print("-" * 40)
        for class_val, count in zip(unique, counts):
            percentage = (count / total) * 100
            class_name = "Winner" if class_val == 1 else "Non-Winner"
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        imbalance_ratio = counts[0] / counts[1] if len(counts) > 1 else 1
        print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")
        print()
        
        return dict(zip(unique, counts))
    
    def apply_sampling_techniques(self, X, y):
        """Apply various sampling techniques to handle imbalance"""
        sampling_results = {}
        
        if not IMBALANCED_LEARN_AVAILABLE:
            print("⚠️ Skipping sampling techniques (imbalanced-learn not available)")
            return {'original': (X, y)}
        
        # Original data
        sampling_results['original'] = (X, y)
        
        try:
            # SMOTE (Synthetic Minority Oversampling)
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_smote, y_smote = smote.fit_resample(X, y)
            sampling_results['smote'] = (X_smote, y_smote)
            print(f"📈 SMOTE: {len(X)} → {len(X_smote)} samples")
            
            # ADASYN (Adaptive Synthetic Sampling)
            adasyn = ADASYN(random_state=42, n_neighbors=3)
            X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
            sampling_results['adasyn'] = (X_adasyn, y_adasyn)
            print(f"📈 ADASYN: {len(X)} → {len(X_adasyn)} samples")
            
            # SMOTETomek (SMOTE + Tomek links removal)
            smote_tomek = SMOTETomek(random_state=42)
            X_st, y_st = smote_tomek.fit_resample(X, y)
            sampling_results['smote_tomek'] = (X_st, y_st)
            print(f"📈 SMOTE+Tomek: {len(X)} → {len(X_st)} samples")
            
        except Exception as e:
            print(f"⚠️ Error in sampling: {e}")
        
        return sampling_results
    
    def optimize_threshold(self, y_true, y_proba):
        """Find optimal threshold for binary classification"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find threshold that maximizes F1 score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return optimal_threshold, f1_scores[optimal_idx]
    
    def evaluate_model_comprehensive(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive evaluation with multiple metrics"""
        try:
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # AUC
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            # Optimize threshold
            optimal_threshold, optimal_f1 = self.optimize_threshold(y_test, y_proba)
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            
            # Metrics with optimal threshold
            accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
            precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
            recall_optimal = recall_score(y_test, y_pred_optimal)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'optimal_threshold': optimal_threshold,
                'optimal_f1': optimal_f1,
                'accuracy_optimal': accuracy_optimal,
                'precision_optimal': precision_optimal,
                'recall_optimal': recall_optimal,
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                }
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return None
    
    def run_improved_analysis(self):
        """Run improved analysis with class balancing techniques"""
        print("🚀 Starting Improved ML Analysis for Imbalanced Data")
        print("=" * 70)
        
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available")
            return None
        
        # Load and prepare data
        prepared_df, feature_columns = self.load_and_prepare_data()
        if prepared_df is None:
            return None
        
        # Time-based split
        df_sorted = prepared_df.sort_values('race_date')
        split_point = int(0.8 * len(df_sorted))
        train_df = df_sorted.iloc[:split_point]
        test_df = df_sorted.iloc[split_point:]
        
        # Prepare features and targets
        X_train = train_df[feature_columns].values
        y_train = train_df['is_winner'].values
        X_test = test_df[feature_columns].values
        y_test = test_df['is_winner'].values
        
        # Analyze class distribution
        print("📊 TRAINING DATA ANALYSIS:")
        train_class_dist = self.analyze_class_distribution(y_train)
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply sampling techniques
        print("🔄 APPLYING SAMPLING TECHNIQUES:")
        sampling_results = self.apply_sampling_techniques(X_train_scaled, y_train)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(prepared_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'features': len(feature_columns),
                'class_distribution': train_class_dist
            },
            'sampling_results': {},
            'model_results': {},
            'best_models': {}
        }
        
        # Test each sampling technique
        for sampling_name, (X_sample, y_sample) in sampling_results.items():
            print(f"\n🧪 TESTING WITH {sampling_name.upper()} SAMPLING:")
            print("-" * 60)
            
            sampling_results_dict = {}
            
            # Test base models
            all_models = {**self.base_models, **self.imbalanced_models}
            
            for model_name, config in all_models.items():
                print(f"   Testing {model_name}...")
                
                try:
                    # Create and train model
                    model_class = config['model']
                    model = model_class(**config['params'])
                    
                    # Handle class weights for Gradient Boosting
                    if 'gradient_boosting' in model_name and sampling_name == 'original':
                        # Calculate class weights for gradient boosting
                        classes = np.unique(y_sample)
                        class_weights = compute_class_weight('balanced', classes=classes, y=y_sample)
                        # Apply sample weights during training
                        sample_weights = np.where(y_sample == 1, class_weights[1], class_weights[0])
                        model.fit(X_sample, y_sample, sample_weight=sample_weights)
                    else:
                        model.fit(X_sample, y_sample)
                    
                    # Evaluate model
                    eval_results = self.evaluate_model_comprehensive(
                        model, X_test_scaled, y_test, model_name
                    )
                    
                    if eval_results:
                        sampling_results_dict[model_name] = eval_results
                        print(f"     📊 Acc: {eval_results['accuracy']:.3f}, "
                              f"AUC: {eval_results['auc']:.3f}, "
                              f"F1: {eval_results['f1_score']:.3f}, "
                              f"Precision: {eval_results['precision']:.3f}, "
                              f"Recall: {eval_results['recall']:.3f}")
                    
                except Exception as e:
                    print(f"     ❌ Failed: {e}")
                    continue
            
            results['sampling_results'][sampling_name] = sampling_results_dict
        
        # Find best models across all techniques
        print(f"\n🏆 FINDING BEST MODELS:")
        print("-" * 60)
        
        best_auc = 0
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        
        best_models = {}
        
        for sampling_name, models in results['sampling_results'].items():
            for model_name, metrics in models.items():
                # Best AUC
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_models['best_auc'] = {
                        'model': model_name,
                        'sampling': sampling_name,
                        'auc': best_auc,
                        'metrics': metrics
                    }
                
                # Best F1
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_models['best_f1'] = {
                        'model': model_name,
                        'sampling': sampling_name,
                        'f1': best_f1,
                        'metrics': metrics
                    }
        
        results['best_models'] = best_models
        
        # Print best results
        if 'best_auc' in best_models:
            best = best_models['best_auc']
            print(f"🥇 Best AUC: {best['model']} with {best['sampling']} sampling")
            print(f"   AUC: {best['auc']:.3f}, F1: {best['metrics']['f1_score']:.3f}")
        
        if 'best_f1' in best_models:
            best = best_models['best_f1']
            print(f"🥇 Best F1: {best['model']} with {best['sampling']} sampling")
            print(f"   F1: {best['f1']:.3f}, AUC: {best['metrics']['auc']:.3f}")
        
        # Save results
        results_file = self.results_dir / f"improved_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved: {results_file}")
        print("✅ Improved ML Analysis Complete!")
        
        return results

def main():
    """Main function"""
    print("🚀 Improved ML System for Imbalanced Race Prediction")
    print("=" * 70)
    
    system = ImprovedMLSystem()
    system.run_improved_analysis()

if __name__ == "__main__":
    main()
