"""
SHAP Explainability Integration Module
====================================

Integrates SHAP (SHapley Additive exPlanations) to provide model explainability.
Builds TreeExplainer or KernelExplainer per base model and caches them.
Exposes get_shap_values(features) returning top-N feature impacts.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP explainer for greyhound racing prediction models."""
    
    def __init__(self, 
                 model_path: str = 'model_registry/models/',
                 db_path: str = "greyhound_racing_data.db",
                 cache_dir: str = 'models/'):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required but not installed. Run: pip install shap")
            
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        self.db_path = db_path
        self.explainers: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔍 Initializing SHAP explainer with models from: {self.model_path}")
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP explainers for all available models."""
        if not self.model_path.exists():
            logger.warning(f"Model path {self.model_path} does not exist")
            return
            
        model_files = list(self.model_path.glob("*.joblib"))
        logger.info(f"Found {len(model_files)} model files")
        
        for model_file in model_files:
            try:
                self._load_or_create_explainer(model_file)
            except Exception as e:
                logger.error(f"Failed to create explainer for {model_file.name}: {e}")
                continue
    
    def _load_or_create_explainer(self, model_file: Path):
        """Load cached explainer or create new one for a model."""
        model_name = model_file.stem
        explainer_cache_path = self.cache_dir / f"shap_explainer_{model_name}.joblib"
        
        # Try to load cached explainer first
        if explainer_cache_path.exists():
            try:
                explainer = joblib.load(explainer_cache_path)
                self.explainers[model_name] = explainer
                logger.info(f"✅ Loaded cached SHAP explainer for {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached explainer for {model_name}: {e}")
        
        # Create new explainer
        logger.info(f"🔨 Creating new SHAP explainer for {model_name}")
        
        try:
            # Load the model
            model = joblib.load(model_file)
            
            # Determine model type and create appropriate explainer
            explainer = self._create_explainer_for_model(model, model_name)
            
            if explainer:
                self.explainers[model_name] = explainer
                # Cache the explainer
                joblib.dump(explainer, explainer_cache_path)
                logger.info(f"✅ Created and cached SHAP explainer for {model_name}")
            else:
                logger.warning(f"Could not create explainer for {model_name}")
                
        except Exception as e:
            logger.error(f"Error creating explainer for {model_name}: {e}")
    
    def _create_explainer_for_model(self, model: Any, model_name: str) -> Optional[Any]:
        """Create appropriate SHAP explainer based on model type."""
        from sklearn.preprocessing import RobustScaler, StandardScaler
        from sklearn.ensemble import (
            GradientBoostingClassifier, 
            RandomForestClassifier, 
            ExtraTreesClassifier
        )
        from sklearn.linear_model import LogisticRegression
        
        # Handle scaler models (these are preprocessing, not prediction models)
        if isinstance(model, (RobustScaler, StandardScaler)):
            logger.debug(f"Skipping scaler model: {model_name}")
            return None
        
        # Tree-based models: Use TreeExplainer (faster and more accurate)
        tree_models = (
            GradientBoostingClassifier,
            RandomForestClassifier, 
            ExtraTreesClassifier
        )
        
        if isinstance(model, tree_models) or hasattr(model, 'feature_importances_'):
            logger.debug(f"Creating TreeExplainer for {model_name}")
            return shap.TreeExplainer(model)
        
        # Linear models: Use LinearExplainer if available
        if isinstance(model, LogisticRegression):
            logger.debug(f"Creating LinearExplainer for {model_name}")
            try:
                return shap.LinearExplainer(model, self._get_background_data())
            except Exception as e:
                logger.warning(f"LinearExplainer failed for {model_name}: {e}")
        
        # Fallback: Use KernelExplainer (slower but works with any model)
        logger.debug(f"Creating KernelExplainer for {model_name}")
        try:
            background_data = self._get_background_data()
            return shap.KernelExplainer(model.predict_proba, background_data)
        except Exception as e:
            logger.warning(f"KernelExplainer failed for {model_name}: {e}")
            return None
    
    def _get_background_data(self, n_samples: int = 50) -> np.ndarray:
        """Get background data for SHAP explainers with efficient fallback."""
        try:
            # Try to load cached background data first
            cache_path = self.cache_dir / "background_data.joblib"
            if cache_path.exists():
                try:
                    background = joblib.load(cache_path)
                    logger.debug(f"Loaded cached background data with shape: {background.shape}")
                    return background
                except Exception as e:
                    logger.warning(f"Failed to load cached background data: {e}")
            
            # Create minimal background data without complex feature engineering
            logger.debug("Creating minimal background data")
            
            # Create sample data representing typical greyhound racing features
            np.random.seed(42)
            n_features = 30  # Estimated number of features
            
            background = np.random.normal(0, 1, (n_samples, n_features))
            
            # Add some realistic racing data patterns
            background[:, 0] = np.random.randint(1, 9, n_samples)  # box_number
            background[:, 1] = np.random.normal(30.5, 2, n_samples)  # weight
            background[:, 2] = np.random.exponential(5, n_samples) + 1  # starting_price
            background[:, 3] = np.random.normal(30, 2, n_samples)  # individual_time
            background[:, 4] = np.random.randint(6, 10, n_samples)  # field_size
            
            # Cache the background data
            try:
                joblib.dump(background, cache_path)
                logger.debug("Cached background data for future use")
            except Exception as e:
                logger.warning(f"Failed to cache background data: {e}")
            
            logger.debug(f"Created background data with shape: {background.shape}")
            return background
            
        except Exception as e:
            logger.error(f"Error creating background data: {e}")
            # Final fallback to minimal dummy data
            return np.zeros((10, 30))
    
    def get_shap_values(self, features: Union[pd.DataFrame, np.ndarray], 
                       model_name: Optional[str] = None, 
                       top_n: int = 10) -> Dict[str, Any]:
        """Get top-N SHAP feature impacts for given features.
        
        Args:
            features: Input features for explanation
            model_name: Specific model to use (if None, use first available)
            top_n: Number of top features to return
            
        Returns:
            Dictionary with explainability information
        """
        if not self.explainers:
            return {
                "error": "No SHAP explainers available",
                "available_models": [],
                "feature_importance": {}
            }
        
        # Select model
        if model_name is None:
            model_name = list(self.explainers.keys())[0]
        
        if model_name not in self.explainers:
            available_models = list(self.explainers.keys())
            return {
                "error": f"Model '{model_name}' not found",
                "available_models": available_models,
                "feature_importance": {}
            }
        
        try:
            explainer = self.explainers[model_name]
            
            # Convert to DataFrame if numpy array
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features)
            
            # Ensure single sample (take first row if multiple)
            if len(features) > 1:
                features = features.iloc[[0]]
            
            # Get SHAP values
            shap_values = explainer.shap_values(features)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class output, take positive class (index 1)
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
            
            # Flatten if needed
            if shap_vals.ndim > 1:
                shap_vals = shap_vals.flatten()
            
            # Get feature names
            feature_names = features.columns.tolist() if hasattr(features, 'columns') else [f"feature_{i}" for i in range(len(shap_vals))]
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, shap_vals))
            
            # Get top N features by absolute importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:top_n]
            
            top_features = dict(sorted_features)
            
            # Calculate total impact
            total_impact = sum(abs(val) for val in shap_vals)
            
            return {
                "model_used": model_name,
                "total_features": len(feature_importance),
                "total_impact": float(total_impact),
                "top_features": {
                    name: {
                        "value": float(value),
                        "abs_value": float(abs(value)),
                        "direction": "positive" if value > 0 else "negative" if value < 0 else "neutral"
                    } for name, value in top_features.items()
                },
                "feature_importance": {k: float(v) for k, v in feature_importance.items()}
            }
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {
                "error": f"SHAP calculation failed: {str(e)}",
                "model_used": model_name,
                "feature_importance": {}
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models with SHAP explainers."""
        return list(self.explainers.keys())
    
    def add_explainability_to_prediction(self, prediction: Dict[str, Any], 
                                        features: Union[pd.DataFrame, np.ndarray],
                                        model_name: Optional[str] = None) -> Dict[str, Any]:
        """Add explainability information to a prediction result.
        
        Args:
            prediction: Existing prediction dictionary
            features: Features used for the prediction
            model_name: Model to use for explanation
            
        Returns:
            Enhanced prediction with explainability
        """
        shap_result = self.get_shap_values(features, model_name)
        prediction["explainability"] = shap_result
        return prediction


# Global instance (lazy loading)
_shap_explainer_instance = None

def get_shap_explainer() -> SHAPExplainer:
    """Get or create global SHAP explainer instance."""
    global _shap_explainer_instance
    if _shap_explainer_instance is None:
        _shap_explainer_instance = SHAPExplainer()
    return _shap_explainer_instance


def get_shap_values(features: Union[pd.DataFrame, np.ndarray], 
                   model_name: Optional[str] = None, 
                   top_n: int = 10) -> Dict[str, Any]:
    """Convenience function to get SHAP values.
    
    Args:
        features: Input features for explanation
        model_name: Specific model to use (if None, use first available) 
        top_n: Number of top features to return
        
    Returns:
        Dictionary with explainability information
    """
    explainer = get_shap_explainer()
    return explainer.get_shap_values(features, model_name, top_n)
