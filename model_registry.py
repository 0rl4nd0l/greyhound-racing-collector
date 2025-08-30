#!/usr/bin/env python3
"""
Model Registry System
====================

This system manages trained ML models, tracks their performance, and automatically
selects the best performing model for predictions. It ensures that the prediction
system always uses the most advanced and accurate model available.

Features:
- Model versioning and metadata tracking
- Performance comparison and ranking
- Automatic best model selection
- Model rollback capabilities
- Thread-safe operations

Author: AI Assistant
Date: July 27, 2025
"""

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""

    model_id: str
    model_name: str
    model_type: str
    training_timestamp: str
    accuracy: float
    auc: float
    f1_score: float
    precision: float
    recall: float
    training_samples: int
    test_samples: int
    features_count: int
    feature_names: List[str]
    model_file_path: str
    scaler_file_path: str
    file_hash: str
    training_duration: float
    hyperparameters: Dict[str, Any]
    validation_method: str
    cross_validation_scores: List[float]
    is_ensemble: bool
    ensemble_components: List[str]
    data_quality_score: float
    model_size_mb: float
    inference_time_ms: float
    is_active: bool = False
    is_best: bool = False
    notes: str = ""
    prediction_type: str = "win"  # Type of prediction (win, place, show, etc.)
    performance_score: float = 0.0  # Composite performance score
    created_at: str = ""  # ISO timestamp when model was created
    # Winner-hit metrics (per-race top-1 correctness)
    correct_winners: int = 0           # Count of races where the model's top pick won
    races_evaluated: int = 0           # Number of races evaluated for the top-1 calculation
    top1_rate: float = 0.0             # correct_winners / races_evaluated (if available)


class ModelRegistry:
    """
    Central registry for managing trained ML models
    """

    def __init__(self, registry_dir: str = "./model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)

        # Core files
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.config_file = self.registry_dir / "registry_config.json"
        self.index_file = self.registry_dir / "model_index.json"

        # Create subdirectories
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()

        # Load existing registry
        self._load_registry()

        logger.info(
            f"🗂️  Model Registry initialized: {len(self.model_index)} models tracked"
        )

    def _load_registry(self):
        """Load existing model registry"""
        try:
            if self.index_file.exists():
                with open(self.index_file, "r") as f:
                    self.model_index = json.load(f)
            else:
                self.model_index = {}

            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    "auto_select_best": True,
                    "max_models_to_keep": 50,
                    # Selection policy: 'performance_score' (default composite), 'auc', 'accuracy', 'f1_score', 'correct_winners'
                    "best_selection_metric": "performance_score",
                    "performance_weight": {
                        "accuracy": 0.4,
                        "auc": 0.3,
                        "f1_score": 0.2,
                        "data_quality": 0.1,
                    },
                }
                self._save_config()

        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            self.model_index = {}
            self.config = {}

    def _save_registry(self):
        """Save model registry to disk"""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.model_index, f, indent=2, default=str)
            logger.debug("📝 Model registry saved")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def _save_config(self):
        """Save registry configuration"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""

    def _calculate_model_score(self, metadata: ModelMetadata) -> float:
        """Calculate composite model score for ranking"""
        weights = self.config.get(
            "performance_weight",
            {"accuracy": 0.4, "auc": 0.3, "f1_score": 0.2, "data_quality": 0.1},
        )

        score = (
            metadata.accuracy * weights.get("accuracy", 0.4)
            + metadata.auc * weights.get("auc", 0.3)
            + metadata.f1_score * weights.get("f1_score", 0.2)
            + metadata.data_quality_score * weights.get("data_quality", 0.1)
        )

        # Bonus for ensemble models
        if metadata.is_ensemble:
            score *= 1.05

        # Penalty for very old models (decay over time)
        try:
            training_date = datetime.fromisoformat(
                metadata.training_timestamp.replace("Z", "+00:00")
            )
            days_old = (datetime.now() - training_date.replace(tzinfo=None)).days
            if days_old > 30:
                score *= max(
                    0.8, 1 - (days_old - 30) * 0.01
                )  # Gradual decay after 30 days
        except Exception:
            pass

        return score

    def register_model(
        self,
        model_obj: Any,
        scaler_obj: Any,
        model_name: str,
        model_type: str,
        performance_metrics: Dict[str, float],
        training_info: Dict[str, Any],
        feature_names: List[str],
        hyperparameters: Dict[str, Any] = None,
        notes: str = "",
    ) -> str:
        """
        Register a new trained model in the registry

        Returns:
            model_id: Unique identifier for the registered model
        """
        with self._lock:
            try:
                # Generate unique model ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_id = f"{model_name}_{model_type}_{timestamp}"

                # Save model and scaler files
                model_file = self.models_dir / f"{model_id}_model.joblib"
                scaler_file = self.models_dir / f"{model_id}_scaler.joblib"

                joblib.dump(model_obj, model_file)
                joblib.dump(scaler_obj, scaler_file)

                # Calculate file hash
                file_hash = self._calculate_file_hash(model_file)

                # Get file size
                model_size_mb = model_file.stat().st_size / (1024 * 1024)

                # Get the created_at timestamp
                created_at = datetime.now().isoformat()
                
                # Create metadata (without performance_score initially)
                metadata = ModelMetadata(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=model_type,
                    training_timestamp=created_at,
                    accuracy=performance_metrics.get("accuracy", 0.0),
                    auc=performance_metrics.get("auc", 0.5),
                    f1_score=performance_metrics.get("f1_score", 0.0),
                    precision=performance_metrics.get("precision", 0.0),
                    recall=performance_metrics.get("recall", 0.0),
                    training_samples=training_info.get("training_samples", 0),
                    test_samples=training_info.get("test_samples", 0),
                    features_count=len(feature_names),
                    feature_names=feature_names,
                    model_file_path=str(model_file),
                    scaler_file_path=str(scaler_file),
                    file_hash=file_hash,
                    training_duration=training_info.get("training_duration", 0.0),
                    hyperparameters=hyperparameters or {},
                    validation_method=training_info.get(
                        "validation_method", "train_test_split"
                    ),
                    cross_validation_scores=training_info.get("cv_scores", []),
                    is_ensemble=training_info.get("is_ensemble", False),
                    ensemble_components=training_info.get("ensemble_components", []),
                    data_quality_score=training_info.get("data_quality_score", 0.5),
                    model_size_mb=model_size_mb,
                    inference_time_ms=training_info.get("inference_time_ms", 0.0),
                    is_active=True,
                    notes=notes,
                    prediction_type=training_info.get("prediction_type", "win"),
                    created_at=created_at,
                    # Winner-hit metrics from training_info or performance_metrics (fallback)
                    correct_winners=int(training_info.get("correct_winners", performance_metrics.get("correct_winners", 0) if isinstance(performance_metrics, dict) else 0)),
                    races_evaluated=int(training_info.get("races_evaluated", performance_metrics.get("races_evaluated", 0) if isinstance(performance_metrics, dict) else 0)),
                    top1_rate=float(training_info.get("top1_rate", performance_metrics.get("top1_rate", 0.0) if isinstance(performance_metrics, dict) else 0.0)),
                    performance_score=0.0  # Will be calculated below
                )
                
                # Calculate and set performance_score using the same logic as _calculate_model_score
                metadata.performance_score = self._calculate_model_score(metadata)

                # Save metadata
                metadata_file = self.metadata_dir / f"{model_id}_metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(asdict(metadata), f, indent=2, default=str)

                # Add to index
                self.model_index[model_id] = asdict(metadata)

                # Update best model if this one is better
                if self.config.get("auto_select_best", True):
                    self._update_best_model()

                self._save_registry()

                logger.info(f"✅ Model registered: {model_id}")
                logger.info(
                    f"   📊 Performance: Acc={metadata.accuracy:.3f}, AUC={metadata.auc:.3f}, F1={metadata.f1_score:.3f}"
                )

                return model_id

            except Exception as e:
                logger.error(f"❌ Error registering model: {e}")
                raise

    def _update_best_model(self):
        """Update the best model designation"""
        try:
            if not self.model_index:
                return

            # Determine selection metric
            selection_metric = (self.config or {}).get("best_selection_metric", "performance_score")

            def _selection_score(md: ModelMetadata) -> float:
                try:
                    if selection_metric == "auc":
                        return float(getattr(md, "auc", 0.0) or 0.0)
                    if selection_metric == "accuracy":
                        return float(getattr(md, "accuracy", 0.0) or 0.0)
                    if selection_metric == "f1_score":
                        return float(getattr(md, "f1_score", 0.0) or 0.0)
                    if selection_metric == "correct_winners":
                        return float(getattr(md, "correct_winners", 0.0) or 0.0)
                    # Default: composite performance score
                    return float(self._calculate_model_score(md))
                except Exception:
                    return 0.0

            # Calculate scores for all active models
            model_scores = []
            for model_id, model_data in self.model_index.items():
                if isinstance(model_data, dict) and model_data.get("is_active", True):
                    try:
                        metadata = ModelMetadata(**model_data)
                        score = _selection_score(metadata)
                        model_scores.append((model_id, score, metadata))
                    except (TypeError, KeyError) as e:
                        logger.warning(f"Error loading metadata for {model_id}: {e}")
                        continue
            if not model_scores:
                return

            # Sort by score (highest first)
            model_scores.sort(key=lambda x: x[1], reverse=True)

            # Clear previous best model flags
            for model_id in self.model_index:
                if isinstance(self.model_index[model_id], dict):
                    self.model_index[model_id]["is_best"] = False

            # Set new best model
            best_model_id, best_score, best_metadata = model_scores[0]
            self.model_index[best_model_id]["is_best"] = True

            # Create/update symlinks to best model
            self._create_best_model_symlinks(best_metadata)

            logger.info(
                f"🏆 Best model updated: {best_model_id} ({selection_metric}: {best_score:.3f})"
            )

        except Exception as e:
            logger.error(f"Error updating best model: {e}")

    def _create_best_model_symlinks(self, best_metadata: ModelMetadata):
        """Create symlinks to the best model for easy access"""
        try:
            # Define symlink paths
            best_model_link = self.registry_dir / "best_model.joblib"
            best_scaler_link = self.registry_dir / "best_scaler.joblib"
            best_metadata_link = self.registry_dir / "best_metadata.json"

            # Remove existing symlinks
            for link in [best_model_link, best_scaler_link, best_metadata_link]:
                if link.exists() or link.is_symlink():
                    link.unlink()

            # Create new symlinks
            model_file = Path(best_metadata.model_file_path)
            scaler_file = Path(best_metadata.scaler_file_path)
            metadata_file = (
                self.metadata_dir / f"{best_metadata.model_id}_metadata.json"
            )

            if model_file.exists():
                best_model_link.symlink_to(model_file.resolve())
            if scaler_file.exists():
                best_scaler_link.symlink_to(scaler_file.resolve())
            if metadata_file.exists():
                best_metadata_link.symlink_to(metadata_file.resolve())

            logger.debug("🔗 Best model symlinks updated")

        except Exception as e:
            logger.warning(f"Could not create best model symlinks: {e}")
            # Fallback: copy files instead of symlinks
            try:
                model_file = Path(best_metadata.model_file_path)
                scaler_file = Path(best_metadata.scaler_file_path)
                metadata_file = (
                    self.metadata_dir / f"{best_metadata.model_id}_metadata.json"
                )

                if model_file.exists():
                    shutil.copy2(model_file, self.registry_dir / "best_model.joblib")
                if scaler_file.exists():
                    shutil.copy2(scaler_file, self.registry_dir / "best_scaler.joblib")
                if metadata_file.exists():
                    shutil.copy2(
                        metadata_file, self.registry_dir / "best_metadata.json"
                    )

                logger.debug("📋 Best model files copied (fallback)")
            except Exception as e2:
                logger.error(f"Could not copy best model files: {e2}")

    def get_best_model(self) -> Optional[Tuple[Any, Any, ModelMetadata]]:
        """
        Load and return the best performing model

        Returns:
            Tuple of (model, scaler, metadata) or None if no models available
        """
        try:
            # Try to load from symlink first (fastest)
            best_model_file = self.registry_dir / "best_model.joblib"
            best_scaler_file = self.registry_dir / "best_scaler.joblib"
            best_metadata_file = self.registry_dir / "best_metadata.json"

            if all(
                f.exists()
                for f in [best_model_file, best_scaler_file, best_metadata_file]
            ):
                try:
                    model = joblib.load(best_model_file)
                    scaler = joblib.load(best_scaler_file)

                    with open(best_metadata_file, "r") as f:
                        metadata_dict = json.load(f)
                    metadata = ModelMetadata(**metadata_dict)

                    return model, scaler, metadata
                except Exception as e:
                    logger.warning(f"Error loading from symlinks: {e}")

            # Fallback: find candidate best models from index, preferring explicit is_best
            candidates: List[Tuple[str, float, ModelMetadata]] = []

            for model_id, model_data in self.model_index.items():
                if not isinstance(model_data, dict):
                    continue
                if not model_data.get("is_active", True):
                    continue
                try:
                    metadata = ModelMetadata(**model_data)
                    score = self._calculate_model_score(metadata)
                    # Boost explicitly marked best to the top by adding a large epsilon
                    if model_data.get("is_best", False):
                        score += 1e6
                    candidates.append((model_id, score, metadata))
                except Exception:
                    continue

            if not candidates:
                logger.warning("No active models found in registry")
                return None

            # Sort by score descending (explicit best first)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Iterate candidates until we find one with existing files
            for model_id, _score, metadata in candidates:
                model_path = Path(metadata.model_file_path)
                scaler_path = Path(metadata.scaler_file_path)
                if not (model_path.exists() and scaler_path.exists()):
                    logger.warning(
                        f"Skipping registry model '{model_id}' due to missing files: "
                        f"model_exists={model_path.exists()}, scaler_exists={scaler_path.exists()}"
                    )
                    continue
                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    return model, scaler, metadata
                except Exception as e:
                    logger.warning(f"Failed loading registry model '{model_id}': {e}")
                    continue

            logger.error("Error loading best model: no valid artifacts found among candidates")
            return None

        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return None

    def get_model_by_id(
        self, model_id: str
    ) -> Optional[Tuple[Any, Any, ModelMetadata]]:
        """Load a specific model by ID"""
        try:
            if model_id not in self.model_index:
                return None

            model_data = self.model_index[model_id]
            metadata = ModelMetadata(**model_data)

            model = joblib.load(metadata.model_file_path)
            scaler = joblib.load(metadata.scaler_file_path)

            return model, scaler, metadata

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None

    def get_best_model_metadata(self) -> Optional[ModelMetadata]:
        """Get metadata for the best model without loading the actual model."""
        try:
            # Find the best model from index
            for model_id, model_data in self.model_index.items():
                if isinstance(model_data, dict) and model_data.get("is_best", False):
                    return ModelMetadata(**model_data)
            
            # Fallback: calculate best model
            candidates = []
            for model_id, model_data in self.model_index.items():
                if isinstance(model_data, dict) and model_data.get("is_active", True):
                    try:
                        metadata = ModelMetadata(**model_data)
                        score = self._calculate_model_score(metadata)
                        candidates.append((model_id, score, metadata))
                    except Exception:
                        continue
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][2]
                
            return None
        except Exception as e:
            logger.error(f"Error getting best model metadata: {e}")
            return None

    def get_most_recent(self, prediction_type: str) -> Optional[ModelMetadata]:
        """Get the most recent model for a given prediction type."""
        models = []
        for model_data in self.model_index.values():
            if (model_data.get("prediction_type") == prediction_type and 
                model_data.get("is_active", True)):
                try:
                    metadata = ModelMetadata(**model_data)
                    models.append(metadata)
                except (TypeError, KeyError) as e:
                    logger.warning(f"Error loading metadata: {e}")
                    continue
        
        if not models:
            return None
        return max(models, key=lambda x: x.created_at)

    def get_best(self, prediction_type: str) -> Optional[ModelMetadata]:
        """Get the best model for a given prediction type based on performance score."""
        models = []
        for model_data in self.model_index.values():
            if (model_data.get("prediction_type") == prediction_type and 
                model_data.get("is_active", True)):
                try:
                    metadata = ModelMetadata(**model_data)
                    models.append(metadata)
                except (TypeError, KeyError) as e:
                    logger.warning(f"Error loading metadata: {e}")
                    continue
        
        if not models:
            return None
        return max(models, key=lambda x: x.performance_score)

    def list_trainable(self, limit: int = 50) -> List[ModelMetadata]:
        """List available trainable models excluding already-running job ids."""
        # Try to import training jobs from the API module
        running_jobs = set()
        try:
            from model_training_api import training_jobs
            running_jobs = {job["model_id"] for job in training_jobs.values() if job["status"] == "running"}
        except ImportError:
            # If we can't import training_jobs, just return all models
            pass
        
        models = []
        for model_data in self.model_index.values():
            if (model_data.get("model_id") not in running_jobs and 
                model_data.get("is_active", True)):
                try:
                    metadata = ModelMetadata(**model_data)
                    models.append(metadata)
                except (TypeError, KeyError) as e:
                    logger.warning(f"Error loading metadata: {e}")
                    continue
        
        # Sort by performance score (descending) then by created_at (recent first)
        models.sort(key=lambda x: (x.performance_score, x.created_at), reverse=True)
        return models[:limit]

    def list_models(self, active_only: bool = True) -> List[ModelMetadata]:
        """List all registered models"""
        models = []
        for model_id, model_data in self.model_index.items():
            if not isinstance(model_data, dict):
                logger.warning(f"Skipping model with invalid data format: {model_id}")
                continue

            if active_only and not model_data.get("is_active", True):
                continue
            try:
                metadata = ModelMetadata(**model_data)
                models.append(metadata)
            except (TypeError, KeyError) as e:
                logger.warning(f"Error loading metadata for {model_id}: {e}")

        # Sort by composite score by default
        models.sort(key=self._calculate_model_score, reverse=True)
        return models

    def set_best_selection_policy(self, metric: str = "auc") -> bool:
        """Set the policy used to auto-select the best model.
        Supported metrics: 'performance_score' (default composite), 'auc', 'accuracy', 'f1_score', 'correct_winners'.
        Returns True if updated.
        """
        with self._lock:
            try:
                metric = str(metric or "").strip().lower()
                if metric not in {"performance_score", "auc", "accuracy", "f1_score", "correct_winners"}:
                    raise ValueError(f"Unsupported metric: {metric}")
                self.config["best_selection_metric"] = metric
                self._save_config()
                # Recompute best based on new policy
                if self.config.get("auto_select_best", True):
                    self._update_best_model()
                return True
            except Exception as e:
                logger.error(f"Failed to set best selection policy: {e}")
                return False

    def auto_promote_best_by_metric(self, metric: str = "auc", prediction_type: Optional[str] = None) -> Optional[str]:
        """Explicitly promote the best model by a specific metric.
        If prediction_type is provided, restrict to that type.
        Returns the promoted model_id or None.
        """
        with self._lock:
            try:
                # Build candidate list
                candidates = []
                for model_id, model_data in self.model_index.items():
                    if not isinstance(model_data, dict):
                        continue
                    if not model_data.get("is_active", True):
                        continue
                    if prediction_type and model_data.get("prediction_type") != prediction_type:
                        continue
                    try:
                        md = ModelMetadata(**model_data)
                        if metric == "auc":
                            score = float(md.auc or 0.0)
                        elif metric == "accuracy":
                            score = float(md.accuracy or 0.0)
                        elif metric == "f1_score":
                            score = float(md.f1_score or 0.0)
                        elif metric == "correct_winners":
                            score = float(md.correct_winners or 0.0)
                        else:
                            score = float(self._calculate_model_score(md))
                        candidates.append((model_id, score, md))
                    except Exception:
                        continue
                if not candidates:
                    return None
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_id, best_score, best_md = candidates[0]
                # Clear flags
                for mid in self.model_index:
                    if isinstance(self.model_index[mid], dict):
                        self.model_index[mid]["is_best"] = False
                # Set new best
                self.model_index[best_id]["is_best"] = True
                self._create_best_model_symlinks(best_md)
                self._save_registry()
                logger.info(f"🏅 Auto-promoted best model by {metric}: {best_id} ({best_score:.3f})")
                return best_id
            except Exception as e:
                logger.error(f"Auto-promote by metric failed: {e}")
                return None

    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate a model (soft delete)"""
        with self._lock:
            if model_id not in self.model_index:
                return False

            self.model_index[model_id]["is_active"] = False
            self.model_index[model_id]["is_best"] = False

            # Update best model if we just deactivated it
            if self.config.get("auto_select_best", True):
                self._update_best_model()

            self._save_registry()
            logger.info(f"🚫 Model deactivated: {model_id}")
            return True

    def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """Permanently delete a model"""
        with self._lock:
            if model_id not in self.model_index:
                return False

            try:
                if remove_files:
                    model_data = self.model_index[model_id]
                    metadata = ModelMetadata(**model_data)

                    # Remove model files
                    for file_path in [
                        metadata.model_file_path,
                        metadata.scaler_file_path,
                    ]:
                        if os.path.exists(file_path):
                            os.remove(file_path)

                    # Remove metadata file
                    metadata_file = self.metadata_dir / f"{model_id}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()

                # Remove from index
                del self.model_index[model_id]

                # Update best model
                if self.config.get("auto_select_best", True):
                    self._update_best_model()

                self._save_registry()
                logger.info(f"🗑️  Model deleted: {model_id}")
                return True

            except Exception as e:
                logger.error(f"Error deleting model {model_id}: {e}")
                return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        active_models = [
            m
            for m in self.model_index.values()
            if isinstance(m, dict) and m.get("is_active", True)
        ]

        if not active_models:
            return {"total_models": 0, "active_models": 0}

        accuracies = [m.get("accuracy", 0) for m in active_models]
        aucs = [m.get("auc", 0.5) for m in active_models]

        best_model = next((m for m in active_models if m.get("is_best", False)), None)

        return {
            "total_models": len(self.model_index),
            "active_models": len(active_models),
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "avg_auc": sum(aucs) / len(aucs) if aucs else 0.5,
            "best_model_id": best_model.get("model_id") if best_model else None,
            "best_model_accuracy": best_model.get("accuracy", 0) if best_model else 0,
            "model_types": list(
                set(m.get("model_type", "unknown") for m in active_models)
            ),
            "registry_size_mb": sum(
                Path(m.get("model_file_path", "")).stat().st_size
                for m in active_models
                if os.path.exists(m.get("model_file_path", ""))
            )
            / (1024 * 1024),
        }

    def cleanup_old_models(self, keep_count: int = None) -> int:
        """Clean up old models, keeping only the best performers"""
        if keep_count is None:
            keep_count = self.config.get("max_models_to_keep", 50)

        active_models = [
            (mid, mdata)
            for mid, mdata in self.model_index.items()
            if mdata.get("is_active", True)
        ]

        if len(active_models) <= keep_count:
            return 0

        # Sort by score and keep the best ones
        model_scores = []
        for model_id, model_data in active_models:
            try:
                metadata = ModelMetadata(**model_data)
                score = self._calculate_model_score(metadata)
                model_scores.append((model_id, score))
            except Exception:
                continue

        model_scores.sort(key=lambda x: x[1], reverse=True)
        models_to_remove = model_scores[keep_count:]

        removed_count = 0
        for model_id, _ in models_to_remove:
            # Don't remove the current best model
            if not self.model_index[model_id].get("is_best", False):
                if self.delete_model(model_id, remove_files=True):
                    removed_count += 1

        logger.info(f"🧹 Cleaned up {removed_count} old models")
        return removed_count


# Global registry instance
_registry_instance = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance (singleton)"""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ModelRegistry()
    return _registry_instance


if __name__ == "__main__":
    # Example usage
    registry = get_model_registry()
    stats = registry.get_registry_stats()
    print(f"📊 Registry Stats: {json.dumps(stats, indent=2)}")

    models = registry.list_models()
    print(f"📋 Found {len(models)} active models")

    best_model = registry.get_best_model()
    if best_model:
        model, scaler, metadata = best_model
        print(f"🏆 Best model: {metadata.model_id} (accuracy: {metadata.accuracy:.3f})")
    else:
        print("❌ No models available")
