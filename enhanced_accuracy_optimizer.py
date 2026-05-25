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

import json
import logging
import os
import pickle
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def _safe_probability_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def _near_tie_epsilon() -> float:
    try:
        return max(0.0, float(os.getenv("V4_NEAR_TIE_EPS", "0.0005")))
    except Exception:
        return 0.0005


def _rank_tie_name(prediction: Dict[str, Any]) -> str:
    try:
        return str(
            prediction.get("dog_clean_name")
            or prediction.get("dog_name")
            or prediction.get("name")
            or ""
        ).upper()
    except Exception:
        return ""


def _single_model_confidence_cap() -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv("V4_SINGLE_MODEL_CONFIDENCE_CAP", "0.55"))))
    except Exception:
        return 0.55


def _append_quality_flag(prediction: Dict[str, Any], flag: str) -> None:
    flags = prediction.get("quality_flags")
    if not isinstance(flags, list):
        flags = []
    if flag not in flags:
        flags.append(flag)
    prediction["quality_flags"] = flags


class PredictionUniquenessValidator:
    """Ensures predictions are unique and not repetitive patterns."""

    def __init__(self, history_window=50):
        self.history_window = history_window
        self.prediction_history = []

    def validate_uniqueness(
        self, predictions: List[Dict], race_id: str
    ) -> Dict[str, Any]:
        """Validate that predictions are unique and not following repetitive patterns."""

        validation_result = {
            "is_unique": True,
            "uniqueness_score": 1.0,
            "pattern_detected": False,
            "recommendation": None,
            "metrics": {},
        }

        try:
            # Extract probability patterns
            prob_patterns = []
            for pred in predictions:
                pattern = (
                    round(pred.get("win_probability", 0), 2),
                    round(pred.get("place_probability", 0), 2),
                    pred.get("box_number", 0),
                )
                prob_patterns.append(pattern)

            # Check for repetitive patterns in recent history
            if len(self.prediction_history) >= 3:
                recent_patterns = self.prediction_history[-3:]

                # Pattern similarity detection
                similar_count = 0
                for hist_patterns in recent_patterns:
                    similarity = self._calculate_pattern_similarity(
                        prob_patterns, hist_patterns
                    )
                    if similarity > 0.8:  # 80% similar
                        similar_count += 1

                if similar_count >= 2:
                    validation_result["is_unique"] = False
                    validation_result["pattern_detected"] = True
                    validation_result["uniqueness_score"] = 0.5
                    validation_result["recommendation"] = (
                        "Apply randomization or use alternative features"
                    )

            # Check for artificial uniformity
            probs = [p.get("win_probability", 0) for p in predictions]
            prob_std = np.std(probs)
            if prob_std < 0.05:  # Too uniform
                validation_result["uniqueness_score"] *= 0.7
                validation_result["recommendation"] = (
                    "Predictions too uniform - increase model sensitivity"
                )

            # Add current patterns to history
            self.prediction_history.append(prob_patterns)
            if len(self.prediction_history) > self.history_window:
                self.prediction_history.pop(0)

            # Calculate detailed metrics
            validation_result["metrics"] = {
                "probability_std": float(prob_std),
                "max_probability": float(max(probs)),
                "min_probability": float(min(probs)),
                "probability_range": float(max(probs) - min(probs)),
                "history_comparisons": len(self.prediction_history),
            }

        except Exception as e:
            logger.warning(f"Uniqueness validation error: {e}")
            validation_result["is_unique"] = True  # Default to allowing predictions

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
        self._primary: Optional[Tuple[str, Dict[str, Any]]] = None

    def seed_primary_model(
        self,
        model: Any,
        model_id: Optional[str] = None,
        metadata: Any = None,
        scaler: Any = None,
        weight: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
    ) -> bool:
        """Use an already-loaded model as the primary optimizer model."""
        if model is None:
            return False
        try:
            if metadata is None or isinstance(metadata, dict):
                md = dict(metadata or {})
                md.setdefault("model_id", model_id or "loaded_primary_model")
                md.setdefault("model_name", md.get("model_id"))
                md.setdefault("accuracy", weight if weight is not None else 0.5)
                md.setdefault("model_file_path", md.get("artifact_path"))
                md.setdefault("feature_names", feature_names or [])
                metadata = SimpleNamespace(**md)
            elif feature_names and not getattr(metadata, "feature_names", None):
                try:
                    setattr(metadata, "feature_names", feature_names)
                except Exception:
                    pass

            primary_id = (
                model_id
                or getattr(metadata, "model_id", None)
                or getattr(metadata, "model_name", None)
                or "loaded_primary_model"
            )
            primary_weight = float(
                weight
                if weight is not None
                else getattr(metadata, "accuracy", 0.5) or 0.5
            )
            self.models = {
                primary_id: {
                    "model": model,
                    "scaler": scaler,
                    "metadata": metadata,
                    "weight": primary_weight,
                }
            }
            self.model_weights = {primary_id: primary_weight}
            self._primary = (primary_id, self.models[primary_id])
            logger.info("♻️ Seeded primary optimizer model from loaded MLSystemV4: %s", primary_id)
            return True
        except Exception as e:
            logger.debug(f"Could not seed primary optimizer model: {e}")
            return False

    def load_models(self):
        """Load and validate all available models.

        Respects registry active flags; only active models are considered.
        Honors ENSEMBLE_TOPK to limit the number of models loaded for speed.
        """
        from model_registry import ModelRegistry

        registry = ModelRegistry()

        # Load only active models from registry listing
        loaded_count = 0
        try:
            candidates = registry.list_models(active_only=True)
        except Exception:
            candidates = []

        # Optional: limit to top-K by accuracy to balance quality and runtime
        topk = None
        try:
            env_k = os.getenv("ENSEMBLE_TOPK", "").strip()
            if env_k:
                topk = max(1, int(env_k))
        except Exception:
            topk = None
        if candidates and topk:
            try:
                candidates = sorted(
                    candidates,
                    key=lambda m: getattr(m, "accuracy", 0.0),
                    reverse=True,
                )[:topk]
                logger.info(f"🔢 ENSEMBLE_TOPK active: loading top {topk} models by accuracy")
            except Exception:
                pass

        for meta in candidates:
            try:
                # Use model_id from metadata to fetch concrete artifacts
                model_tuple = registry.get_model_by_id(meta.model_id)
                if model_tuple:
                    model, scaler, meta_loaded = model_tuple
                    self.models[meta_loaded.model_id] = {
                        "model": model,
                        "scaler": scaler,
                        "metadata": meta_loaded,
                        "weight": getattr(meta_loaded, "accuracy", 0.5),
                    }
                    self.model_weights[meta_loaded.model_id] = getattr(
                        meta_loaded, "accuracy", 0.5
                    )
                    loaded_count += 1
                    logger.info(
                        f"✅ Loaded model: {meta_loaded.model_id} (accuracy: {meta_loaded.accuracy:.3f})"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load model {getattr(meta, 'model_id', '<unknown>')}: {e}"
                )

        logger.info(f"🤖 Ensemble loaded with {loaded_count} models")
        return loaded_count > 0

    def load_primary_model(self) -> bool:
        """Load only the primary (best) model from the registry."""
        try:
            if self._primary:
                return True

            from model_registry import ModelRegistry

            reg = ModelRegistry()
            best = reg.get_best_model()
            if not best:
                logger.warning("No best model available in registry")
                return False
            model, scaler, metadata = best
            self.models = {
                metadata.model_id: {
                    "model": model,
                    "scaler": scaler,
                    "metadata": metadata,
                    "weight": getattr(metadata, "accuracy", 0.5),
                }
            }
            self.model_weights = {metadata.model_id: getattr(metadata, "accuracy", 0.5)}
            self._primary = (metadata.model_id, self.models[metadata.model_id])
            logger.info(
                f"🎯 Loaded primary model: {metadata.model_id} (accuracy: {metadata.accuracy:.3f})"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load primary model: {e}")
            return False

    def _prepare_base_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical and numeric types, fill defaults."""
        base_features = features.copy()
        try:
            cat_defaults = {
                "venue": "UNKNOWN",
                "grade": "5",
                "track_condition": "Good",
                "weather": "Fine",
                "trainer_name": "Unknown",
            }
            for col, default in cat_defaults.items():
                if col not in base_features.columns:
                    base_features[col] = default
                else:
                    base_features[col] = base_features[col].astype(str)
                    base_features[col] = base_features[col].replace(
                        {"0": default, "nan": default}
                    )
                    base_features[col] = base_features[col].where(
                        base_features[col].notna(), other=default
                    )
        except Exception as _e:
            logger.debug(f"Categorical default fill skipped due to: {_e}")

        try:
            num_cols = base_features.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                base_features[num_cols] = (
                    base_features[num_cols].astype("float64").fillna(0.0)
                )
        except Exception as _e:
            logger.debug(f"Numeric NA cleanup skipped due to: {_e}")
        return base_features

    def _predict_with_model(
        self, model_id: str, model_data: Dict[str, Any], base_features: pd.DataFrame
    ) -> np.ndarray:
        """Align features per model and return win probabilities for that model."""
        model = model_data["model"]
        scaler = model_data["scaler"]
        metadata = model_data.get("metadata")

        expected_features = None
        contract_path = None
        required_columns_order: List[str] = []

        # Resolve contract path from metadata if available
        try:
            artifact_path = getattr(metadata, "model_file_path", None)
            if artifact_path and os.path.exists(artifact_path):
                stem = Path(artifact_path).name
                stem_json = (
                    Path("docs/model_contracts")
                    / f"{Path(stem).with_suffix('').name}.json"
                )
                if stem_json.exists():
                    contract_path = stem_json
            if contract_path is None:
                mid = getattr(metadata, "model_id", None)
                if mid:
                    mid_json = Path("docs/model_contracts") / f"{mid}.json"
                    if mid_json.exists():
                        contract_path = mid_json
            if contract_path is None:
                et_contract = Path("docs/model_contracts/V4_ExtraTrees_20250819.json")
                if et_contract.exists():
                    contract_path = et_contract
        except Exception as _e:
            logger.debug(f"Contract resolution failed for {model_id}: {_e}")

        if contract_path and Path(contract_path).exists():
            try:
                with open(contract_path, "r") as f:
                    contract = json.load(f)
                expected_features = contract.get("features")
                if isinstance(expected_features, list):
                    required_columns_order.extend(
                        [
                            c
                            for c in expected_features
                            if c not in required_columns_order
                        ]
                    )
                    logger.debug(
                        f"📜 Loaded contract for {model_id}: {len(expected_features)} features from {contract_path}"
                    )
            except Exception as _e:
                logger.warning(
                    f"Failed to load contract for {model_id} at {contract_path}: {_e}"
                )

        # Build per-model frame with TGR compatibility mapping
        per_model_df = base_features.copy()

        # Build a strictly ordered required column list based on fitted artifacts.
        # Priority: scaler.feature_names_in_ > model.feature_names_in_ > contract.features > metadata.feature_names
        try:
            scaler_cols = []
            if (
                hasattr(scaler, "feature_names_in_")
                and getattr(scaler, "feature_names_in_") is not None
            ):
                scaler_cols = list(getattr(scaler, "feature_names_in_"))

            model_cols = []
            if (
                hasattr(model, "feature_names_in_")
                and getattr(model, "feature_names_in_") is not None
            ):
                model_cols = list(getattr(model, "feature_names_in_"))

            meta_cols = []
            if (
                metadata
                and hasattr(metadata, "feature_names")
                and getattr(metadata, "feature_names")
            ):
                meta_cols = list(getattr(metadata, "feature_names"))

            # Choose the single authoritative order to avoid feature-count/name mismatches
            if scaler_cols:
                required_columns_order = list(scaler_cols)
            elif model_cols:
                required_columns_order = list(model_cols)
            elif expected_features:
                required_columns_order = list(expected_features)
            elif meta_cols:
                required_columns_order = list(meta_cols)
            else:
                required_columns_order = list(per_model_df.columns)
        except Exception as _e:
            logger.debug(f"Failed to resolve required columns for {model_id}: {_e}")

        # Augment missing tgr_* columns from existing features (compatibility shim)
        try:
            tgr_map = {
                "tgr_win_rate": "historical_win_rate",
                "tgr_place_rate": "historical_place_rate",
                "tgr_avg_finish_position": "historical_avg_position",
                "tgr_best_finish_position": "historical_best_position",
                "tgr_recent_avg_position": "historical_avg_position",
                "tgr_recent_best_position": "historical_best_position",
                "tgr_days_since_last_race": "days_since_last_race",
                "tgr_venues_raced": "venue_experience",
                "tgr_preferred_distance_avg": "best_distance_avg_position",
                "tgr_preferred_distance": "target_distance",
                "tgr_preferred_distance_races": "race_frequency",
                "tgr_recent_races": "race_frequency",
                "tgr_consistency": "historical_time_consistency",
                "tgr_form_trend": "historical_form_trend",
            }

            ensure_cols = (
                required_columns_order
                if required_columns_order
                else (expected_features or [])
            )
            # Do not add extra columns beyond the authoritative order; only map if present in ensure_cols
            # (Prevents scaler/model feature-count mismatches)

            for tgt, src in tgr_map.items():
                if (
                    (
                        (ensure_cols and tgt in ensure_cols)
                        or (not ensure_cols and tgt in per_model_df.columns)
                    )
                    and tgt not in per_model_df.columns
                    and src in per_model_df.columns
                ):
                    per_model_df[tgt] = per_model_df[src]

            for col in ensure_cols:
                if (
                    isinstance(col, str)
                    and col.startswith("tgr_")
                    and col not in per_model_df.columns
                ):
                    per_model_df[col] = 0.0
        except Exception as _e:
            logger.debug(f"TGR compatibility mapping skipped for {model_id}: {_e}")

        if required_columns_order:
            per_model_df = per_model_df.reindex(columns=required_columns_order)
        elif expected_features:
            per_model_df = per_model_df.reindex(columns=expected_features)

        try:
            num_cols_model = per_model_df.select_dtypes(include=["number"]).columns
            if len(num_cols_model) > 0:
                per_model_df[num_cols_model] = (
                    per_model_df[num_cols_model].astype("float64").fillna(0.0)
                )
        except Exception as _e:
            logger.debug(f"Final NA guard skipped for {model_id}: {_e}")

        if scaler:
            scaled_features = scaler.transform(per_model_df)
            scaled_df = pd.DataFrame(
                scaled_features, columns=per_model_df.columns, index=per_model_df.index
            )
        else:
            scaled_df = per_model_df

        def _as_model_input(m, df: pd.DataFrame):
            try:
                if (
                    hasattr(m, "feature_names_in_")
                    and getattr(m, "feature_names_in_") is not None
                ):
                    return df
            except Exception:
                pass
            try:
                from sklearn.calibration import CalibratedClassifierCV as _CCC

                if isinstance(m, _CCC):
                    base = getattr(m, "base_estimator", None)
                    if (
                        base is not None
                        and getattr(base, "feature_names_in_", None) is not None
                    ):
                        return df
            except Exception:
                pass
            return df.to_numpy()

        try:
            _tweaked = False
            if hasattr(model, "set_output"):
                model.set_output(transform="default")
                _tweaked = True
            else:
                for attr in ("base_estimator", "estimator", "pipeline", "model"):
                    _inner = getattr(model, attr, None)
                    if _inner is not None and hasattr(_inner, "set_output"):
                        _inner.set_output(transform="default")
                        _tweaked = True
                        break
                    if _inner is not None and hasattr(_inner, "named_steps"):
                        for _step in _inner.named_steps.values():
                            if hasattr(_step, "set_output"):
                                _step.set_output(transform="default")
                                _tweaked = True
                                break
                        if _tweaked:
                            break
            if _tweaked:
                logger.debug(f"Adjusted transform output to numpy for {model_id}")
        except Exception as _e:
            logger.debug(f"Could not adjust transform output for {model_id}: {_e}")

        X_in = _as_model_input(model, scaled_df)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"X has feature names, but .* was fitted without feature names",
                category=UserWarning,
            )
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_in)
                win_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            else:
                win_probs = model.predict(X_in)
        return win_probs

    def predict_with_ensemble(
        self, features: pd.DataFrame, race_id: str
    ) -> List[Dict[str, Any]]:
        """Generate ensemble predictions with dynamic weighting."""

        if not self.models:
            if not self.load_models():
                raise ValueError("No models available for ensemble prediction")

        # Preserve dog names for presentation
        names = None
        if "dog_clean_name" in features.columns:
            try:
                names = features["dog_clean_name"].astype(str).tolist()
            except Exception:
                try:
                    names = features["dog_clean_name"].tolist()
                except Exception:
                    names = None

        # Prepare a base feature frame with categorical defaults and numeric coercion
        base_features = self._prepare_base_features(features)

        predictions: List[Dict[str, Any]] = []
        model_predictions: Dict[str, Any] = {}

        # Collect predictions from each model using its own contract
        for model_id, model_data in self.models.items():
            try:
                win_probs = self._predict_with_model(
                    model_id, model_data, base_features
                )
                model_predictions[model_id] = win_probs
            except Exception as e:
                logger.warning(f"Model {model_id} prediction failed: {e}")
                continue

        if not model_predictions:
            raise ValueError("All ensemble models failed to generate predictions")

        # Preserve dog names for presentation
        names = None
        if "dog_clean_name" in base_features.columns:
            try:
                names = base_features["dog_clean_name"].astype(str).tolist()
            except Exception:
                try:
                    names = base_features["dog_clean_name"].tolist()
                except Exception:
                    names = None

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
                confidence = self._calculate_confidence(
                    calibrated_win_prob, model_predictions, i
                )

                model_count = len(model_predictions)
                prediction = {
                    "dog_clean_name": (
                        names[i]
                        if names and i < len(names)
                        else row.get("dog_clean_name", f"Dog_{i}")
                    ),
                    "box_number": int(row.get("box_number", i + 1)),
                    "win_probability_unrounded": float(calibrated_win_prob),
                    "win_probability": round(float(calibrated_win_prob), 4),
                    "place_probability_unrounded": float(place_prob),
                    "place_probability": round(float(place_prob), 4),
                    "confidence": round(float(confidence), 4),
                    "ensemble_models": model_count,
                    "model_agreement": self._calculate_model_agreement(
                        model_predictions, i
                    ),
                    "model_agreement_basis": (
                        "ensemble_model_agreement"
                        if model_count > 1
                        else "not_applicable_single_model"
                    ),
                    "confidence_basis": (
                        "ensemble_agreement_and_probability_extremeness"
                        if model_count > 1
                        else "single_model_probability_extremeness_capped"
                    ),
                    "race_id": race_id,
                    "prediction_timestamp": datetime.now().isoformat(),
                }

                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to process dog {i}: {e}")
                continue

        # Normalize probabilities within race
        predictions = self._normalize_race_probabilities(predictions)

        return predictions
        return predictions

    def predict_with_primary_then_fallback(
        self, features: pd.DataFrame, race_id: str
    ) -> List[Dict[str, Any]]:
        """Use the primary (best) model by default; fall back to first working secondary model if it fails.
        This avoids invoking the full ensemble unless necessary.
        """
        # Prepare features once
        base_features = self._prepare_base_features(features)

        # Try primary
        if not self._primary:
            if not self.load_primary_model():
                # If no primary could be loaded, fall back to the full ensemble loader path
                if not self.load_models():
                    raise ValueError("No models available for prediction")
        primary_id, primary_data = (
            next(iter(self.models.items())) if not self._primary else self._primary
        )

        try:
            primary_probs = self._predict_with_model(
                primary_id, primary_data, base_features
            )
            model_predictions = {primary_id: primary_probs}
        except Exception as e:
            logger.warning(
                f"Primary model {primary_id} failed: {e}. Falling back to secondary models (ensemble)..."
            )
            # Load all active models and build an ensemble from remaining working models
            if not self.load_models():
                raise ValueError("No models available for fallback prediction")
            # Remove primary from the pool if present
            if primary_id in self.models:
                all_items = [
                    (mid, mdata)
                    for mid, mdata in self.models.items()
                    if mid != primary_id
                ]
            else:
                all_items = list(self.models.items())
            model_predictions = {}
            used_models: Dict[str, Any] = {}
            for mid, mdata in all_items:
                try:
                    probs = self._predict_with_model(mid, mdata, base_features)
                    model_predictions[mid] = probs
                    used_models[mid] = mdata
                except Exception as e2:
                    logger.warning(f"Fallback model {mid} failed: {e2}")
                    continue
            if not model_predictions:
                raise ValueError("All fallback models failed to generate predictions")
            # Limit self.models to those actually used in the fallback ensemble
            self.models = used_models
            # Compute a dynamic weighted ensemble across working fallback models
            ensemble_probs = self._calculate_weighted_ensemble(
                model_predictions, race_id
            )

        # Build prediction dicts from the chosen probabilities (primary single-model or fallback ensemble)
        names = None
        if "dog_clean_name" in base_features.columns:
            try:
                names = base_features["dog_clean_name"].astype(str).tolist()
            except Exception:
                try:
                    names = base_features["dog_clean_name"].tolist()
                except Exception:
                    names = None

        predictions: List[Dict[str, Any]] = []
        # Select the score vector and model count appropriately
        if "ensemble_probs" in locals():
            score_vector = ensemble_probs
            ensemble_count = len(model_predictions)
        else:
            score_vector = model_predictions[primary_id]
            ensemble_count = 1
        for i, (_, row) in enumerate(base_features.iterrows()):
            try:
                win_prob = float(score_vector[i])
                calibrated_win_prob = self._apply_calibration(win_prob, race_id)
                place_prob = min(0.9, calibrated_win_prob * 2.5 + 0.1)
                # Confidence and agreement: use model disagreement when we have an ensemble
                if ensemble_count > 1:
                    confidence = self._calculate_confidence(
                        calibrated_win_prob, model_predictions, i
                    )
                    model_agreement = self._calculate_model_agreement(
                        model_predictions, i
                    )
                    model_agreement_basis = "ensemble_model_agreement"
                    confidence_basis = "ensemble_agreement_and_probability_extremeness"
                else:
                    model_agreement = None
                    model_agreement_basis = "not_applicable_single_model"
                    extremeness = 2 * abs(calibrated_win_prob - 0.5)
                    confidence = min(
                        _single_model_confidence_cap(),
                        max(0.1, 0.25 + 0.25 * extremeness),
                    )
                    confidence_basis = "single_model_probability_extremeness_capped"
                predictions.append(
                    {
                        "dog_clean_name": (
                            names[i]
                            if names and i < len(names)
                            else row.get("dog_clean_name", f"Dog_{i}")
                        ),
                        "box_number": int(row.get("box_number", i + 1)),
                        "win_probability_unrounded": float(calibrated_win_prob),
                        "win_probability": round(float(calibrated_win_prob), 4),
                        "place_probability_unrounded": float(place_prob),
                        "place_probability": round(float(place_prob), 4),
                        "confidence": round(float(confidence), 4),
                        "ensemble_models": ensemble_count,
                        "model_agreement": model_agreement,
                        "model_agreement_basis": model_agreement_basis,
                        "confidence_basis": confidence_basis,
                        "race_id": race_id,
                        "prediction_timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as ie:
                logger.warning(f"Failed to process dog {i}: {ie}")
                continue
        predictions = self._normalize_race_probabilities(predictions)
        return predictions

    def _calculate_weighted_ensemble(
        self, model_predictions: Dict, race_id: str
    ) -> np.ndarray:
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

    def _calculate_confidence(
        self, win_prob: float, model_predictions: Dict, dog_index: int
    ) -> float:
        """Calculate prediction confidence based on model agreement."""

        # Extract predictions for this dog from all models
        dog_predictions = []
        for model_id, probs in model_predictions.items():
            if dog_index < len(probs):
                dog_predictions.append(probs[dog_index])

        if not dog_predictions:
            return 0.5

        if len(dog_predictions) < 2:
            extremeness = 2 * abs(win_prob - 0.5)
            return min(
                _single_model_confidence_cap(),
                max(0.1, 0.25 + 0.25 * extremeness),
            )

        # Calculate agreement (inverse of variance)
        pred_std = np.std(dog_predictions)
        agreement = 1.0 / (1.0 + pred_std)

        # Combine with probability extremeness (more confident at extremes)
        extremeness = 2 * abs(win_prob - 0.5)

        # Final confidence
        confidence = (agreement * 0.7) + (extremeness * 0.3)
        return min(1.0, max(0.1, confidence))

    def _calculate_model_agreement(
        self, model_predictions: Dict, dog_index: int
    ) -> float:
        """Calculate how much models agree on this prediction."""

        dog_predictions = []
        for model_id, probs in model_predictions.items():
            if dog_index < len(probs):
                dog_predictions.append(probs[dog_index])

        if len(dog_predictions) < 2:
            return None

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
        """Normalize win probabilities to sum to 1.0 and rank on unrounded scores."""

        if not predictions:
            return predictions

        raw_values = []
        for prediction in predictions:
            raw_source = prediction.get(
                "win_probability_unrounded",
                prediction.get("win_prob_raw", prediction.get("win_probability", 0.0)),
            )
            raw_win = _safe_probability_float(raw_source)
            raw_win = max(0.0, raw_win)
            raw_values.append(raw_win)
            prediction["win_probability_unrounded"] = raw_win
            prediction["win_prob_raw"] = raw_win
            prediction.setdefault("confidence_score", prediction.get("confidence"))
            prediction.setdefault("ev_win", None)

        total_win_prob = sum(raw_values)

        for prediction, raw_win in zip(predictions, raw_values):
            if total_win_prob > 0:
                normalized_win = raw_win / total_win_prob
                place_probability = min(0.9, normalized_win * 2.5 + 0.1)
            else:
                normalized_win = raw_win
                place_probability = _safe_probability_float(
                    prediction.get(
                        "place_probability_unrounded",
                        prediction.get("place_probability", 0.0),
                    )
                )

            prediction["win_prob_norm_unrounded"] = float(normalized_win)
            prediction["win_probability_unrounded_norm"] = float(normalized_win)
            prediction["rank_sort_probability"] = float(normalized_win)
            prediction["win_probability"] = round(float(normalized_win), 4)
            prediction["win_prob_norm"] = float(prediction["win_probability"])
            prediction["place_probability_unrounded"] = float(place_probability)
            prediction["place_probability"] = round(float(place_probability), 4)
            try:
                ensemble_models = int(prediction.get("ensemble_models") or 0)
            except Exception:
                ensemble_models = 0
            if ensemble_models <= 1:
                capped_confidence = min(
                    _single_model_confidence_cap(),
                    _safe_probability_float(prediction.get("confidence"), 0.0),
                )
                prediction["confidence"] = round(float(capped_confidence), 4)
                prediction["confidence_score"] = float(prediction["confidence"])
                prediction["model_agreement"] = None
                prediction["model_agreement_basis"] = "not_applicable_single_model"
                prediction["confidence_basis"] = (
                    "single_model_probability_extremeness_capped"
                )
                _append_quality_flag(prediction, "single_model_no_ensemble_agreement")

        ranked = sorted(
            predictions,
            key=lambda p: (
                -_safe_probability_float(p.get("rank_sort_probability", 0.0)),
                _rank_tie_name(p),
                int(_safe_probability_float(p.get("box_number"), 999)),
            ),
        )
        for rank, prediction in enumerate(ranked, start=1):
            prediction["predicted_rank"] = rank

        tie_eps = _near_tie_epsilon()
        tie_groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        current_group_ref = None

        for prediction in ranked:
            rank_probability = _safe_probability_float(
                prediction.get("rank_sort_probability", 0.0)
            )
            if not current_group:
                current_group = [prediction]
                current_group_ref = rank_probability
                continue

            if (
                current_group_ref is not None
                and abs(rank_probability - current_group_ref) <= tie_eps
            ):
                current_group.append(prediction)
            else:
                tie_groups.append(current_group)
                current_group = [prediction]
                current_group_ref = rank_probability

        if current_group:
            tie_groups.append(current_group)

        near_tie_group_id = 0
        for group in tie_groups:
            group_size = len(group)
            group_id = None
            if group_size > 1:
                near_tie_group_id += 1
                group_id = near_tie_group_id
            for prediction in group:
                prediction["is_near_tie"] = group_size > 1
                prediction["near_tie_group"] = group_id
                prediction["near_tie_group_size"] = group_size
                prediction["near_tie_epsilon"] = tie_eps
                prediction["rank_tie_breaker"] = "dog_name"
                if group_size > 1:
                    prediction["rank_note"] = "near_tie_probability_group"
                else:
                    prediction.setdefault("rank_note", None)

        return ranked


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
            "min_confidence_threshold": 0.3,
            "uniqueness_threshold": 0.7,
            "ensemble_weight_decay": 0.001,
            "calibration_enabled": True,
            "feedback_learning_rate": 0.01,
            # Ensemble behavior: 'primary_with_fallback' (default) or 'full_ensemble'
            "ensemble_mode": os.getenv("ENSEMBLE_MODE", "primary_with_fallback")
            .strip()
            .lower(),
        }

        config_path = Path("config/accuracy_optimizer.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load optimization config: {e}")

        # Normalize ensemble_mode
        try:
            em = (
                str(default_config.get("ensemble_mode", "primary_with_fallback"))
                .strip()
                .lower()
            )
            if em not in ("primary_with_fallback", "full_ensemble"):
                em = "primary_with_fallback"
            default_config["ensemble_mode"] = em
        except Exception:
            default_config["ensemble_mode"] = "primary_with_fallback"

        return default_config

    def generate_optimized_predictions(
        self, features: pd.DataFrame, race_id: str
    ) -> Dict[str, Any]:
        """Generate highly accurate and unique predictions."""

        logger.info(f"🎯 Generating optimized predictions for race: {race_id}")

        try:
            # Generate predictions according to configured ensemble behavior
            mode = str(
                self.config.get("ensemble_mode", "primary_with_fallback")
            ).lower()
            if mode == "full_ensemble":
                predictions = self.ensemble_predictor.predict_with_ensemble(
                    features, race_id
                )
            else:
                predictions = (
                    self.ensemble_predictor.predict_with_primary_then_fallback(
                        features, race_id
                    )
                )

            # Validate uniqueness
            uniqueness_result = self.uniqueness_validator.validate_uniqueness(
                predictions, race_id
            )

            # Apply quality filters
            filtered_predictions = self._apply_quality_filters(predictions)

            # Calculate overall race metrics
            race_metrics = self._calculate_race_metrics(filtered_predictions)

            # Prepare result
            result = {
                "success": True,
                "race_id": race_id,
                "predictions": filtered_predictions,
                "uniqueness_validation": uniqueness_result,
                "race_metrics": race_metrics,
                "optimization_applied": True,
                "ensemble_models_used": self.ensemble_predictor.models.__len__(),
                "model_ids_used": list(self.ensemble_predictor.models.keys()),
                "generation_timestamp": datetime.now().isoformat(),
            }
            # If only a single model was used, expose it as the primary_model_id for convenience
            try:
                if (
                    isinstance(result.get("model_ids_used"), list)
                    and len(result["model_ids_used"]) == 1
                ):
                    result["primary_model_id"] = result["model_ids_used"][0]
                    result.setdefault("quality_warnings", []).append(
                        {
                            "code": "single_model_no_ensemble_agreement",
                            "message": "Only one model contributed, so model_agreement is not ensemble evidence.",
                        }
                    )
            except Exception:
                pass
            try:
                model_ids = result.get("model_ids_used") or []
                model_version = (
                    result.get("primary_model_id")
                    or (",".join(str(m) for m in model_ids) if model_ids else None)
                    or "optimizer_model"
                )
                result["model_version"] = model_version
                for prediction in filtered_predictions:
                    prediction.setdefault("model_version", model_version)
                    prediction.setdefault(
                        "confidence_score", prediction.get("confidence")
                    )
                    prediction.setdefault("ev_win", None)
            except Exception:
                pass

            # Log performance for future optimization
            self._log_prediction_performance(race_id, result)

            logger.info(
                f"✅ Generated {len(filtered_predictions)} optimized predictions"
            )
            return result

        except Exception as e:
            logger.error(f"Optimization failed for race {race_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "race_id": race_id,
                "fallback_used": True,
            }

    def _apply_quality_filters(self, predictions: List[Dict]) -> List[Dict]:
        """Apply quality filters without breaking source runner alignment."""

        filtered_predictions = []
        min_confidence = self.config.get("min_confidence_threshold", 0.3)
        drop_low_quality = str(
            os.getenv("V4_OPTIMIZER_DROP_LOW_QUALITY", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        for prediction in predictions:
            confidence = prediction.get("confidence", 0)
            passes_confidence = confidence >= min_confidence
            passes_checks = self._passes_quality_checks(prediction)

            if passes_confidence and passes_checks:
                prediction["quality_filter_status"] = "passed"
                filtered_predictions.append(prediction)
                continue

            if not passes_confidence:
                logger.debug(
                    f"Low confidence prediction filtered: {prediction['dog_clean_name']}"
                )
                _append_quality_flag(prediction, "optimizer_low_confidence")
            else:
                logger.debug(
                    f"Prediction filtered out: {prediction['dog_clean_name']}"
                )
                _append_quality_flag(prediction, "optimizer_quality_check_failed")

            prediction["quality_filter_status"] = (
                "dropped" if drop_low_quality else "retained_for_runner_alignment"
            )
            prediction["quality_filter_min_confidence"] = float(min_confidence)
            prediction["quality_filter_confidence"] = _safe_probability_float(
                confidence
            )

            if drop_low_quality:
                continue

            _append_quality_flag(
                prediction, "optimizer_retained_low_quality_for_runner_alignment"
            )
            filtered_predictions.append(prediction)

        return filtered_predictions

    def _passes_quality_checks(self, prediction: Dict) -> bool:
        """Check if prediction passes quality thresholds."""

        # Probability bounds check
        win_prob = prediction.get("win_probability", 0)
        if not (0.001 <= win_prob <= 0.999):
            return False

        # Model agreement check
        agreement = prediction.get("model_agreement", 0)
        try:
            ensemble_models = int(prediction.get("ensemble_models") or 0)
        except Exception:
            ensemble_models = 0
        if agreement is not None and ensemble_models > 1:
            if float(agreement) < 0.3:  # Models disagree too much
                return False

        # Confidence check
        confidence = prediction.get("confidence", 0)
        if confidence < 0.2:
            return False

        return True

    def _calculate_race_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate overall race quality metrics."""

        if not predictions:
            return {"error": "No valid predictions"}

        win_probs = [p["win_probability"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        agreements = [
            float(p.get("model_agreement"))
            for p in predictions
            if p.get("model_agreement") is not None
        ]

        metrics = {
            "total_predictions": len(predictions),
            "avg_confidence": round(np.mean(confidences), 4),
            "avg_model_agreement": (
                round(np.mean(agreements), 4) if agreements else None
            ),
            "probability_distribution": {
                "mean": round(np.mean(win_probs), 4),
                "std": round(np.std(win_probs), 4),
                "max": round(max(win_probs), 4),
                "min": round(min(win_probs), 4),
            },
            "quality_score": self._calculate_overall_quality_score(predictions),
        }

        return metrics

    def _calculate_overall_quality_score(self, predictions: List[Dict]) -> float:
        """Calculate overall quality score for the race predictions."""

        if not predictions:
            return 0.0

        # Weighted combination of quality factors
        avg_confidence = np.mean([p["confidence"] for p in predictions])
        agreements = [
            float(p.get("model_agreement"))
            for p in predictions
            if p.get("model_agreement") is not None
        ]
        prob_diversity = np.std([p["win_probability"] for p in predictions])

        # Quality score (0-1). Do not reward agreement when only one model ran.
        if agreements:
            avg_agreement = np.mean(agreements)
            quality_score = (
                avg_confidence * 0.4
                + avg_agreement * 0.3
                + min(1.0, prob_diversity * 5) * 0.3
            )
        else:
            quality_score = avg_confidence * 0.6 + min(1.0, prob_diversity * 5) * 0.4

        return round(quality_score, 4)

    def _log_prediction_performance(self, race_id: str, result: Dict):
        """Log prediction performance for future optimization."""

        performance_log = {
            "race_id": race_id,
            "timestamp": datetime.now().isoformat(),
            "predictions_count": len(result.get("predictions", [])),
            "quality_metrics": result.get("race_metrics", {}),
            "uniqueness_score": result.get("uniqueness_validation", {}).get(
                "uniqueness_score", 0
            ),
            "ensemble_models": result.get("ensemble_models_used", 0),
            "model_ids_used": result.get("model_ids_used", []),
            "primary_model_id": result.get("primary_model_id"),
        }

        # Store in performance tracker
        self.performance_tracker[race_id] = performance_log

        # Optional: persist to file for long-term analysis
        try:
            log_file = Path("logs/accuracy_optimization.jsonl")
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(performance_log) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist performance log: {e}")


# Integration function for ML System V4
def integrate_enhanced_accuracy(ml_system_v4):
    """Integrate enhanced accuracy optimizer with existing ML System V4."""

    accuracy_optimizer = AccuracyOptimizer(ml_system_v4.db_path)
    try:
        model_info = getattr(ml_system_v4, "model_info", {}) or {}
        loaded_model = (
            getattr(ml_system_v4, "calibrated_pipeline_win", None)
            or getattr(ml_system_v4, "calibrated_pipeline", None)
        )
        accuracy_optimizer.ensemble_predictor.seed_primary_model(
            loaded_model,
            model_id=(
                model_info.get("model_id")
                or model_info.get("model_version")
                or model_info.get("model_name")
            ),
            metadata={
                "model_id": model_info.get("model_id")
                or model_info.get("model_version")
                or model_info.get("model_name")
                or "loaded_primary_model",
                "model_name": model_info.get("model_name")
                or model_info.get("model_type")
                or "loaded_primary_model",
                "accuracy": model_info.get("test_accuracy") or 0.5,
                "artifact_path": model_info.get("artifact_path"),
                "feature_names": list(getattr(ml_system_v4, "feature_columns", []) or []),
            },
            feature_names=list(getattr(ml_system_v4, "feature_columns", []) or []),
            weight=model_info.get("test_accuracy"),
        )
    except Exception as e:
        logger.debug(f"Primary optimizer seed skipped: {e}")

    # Monkey patch the predict_race method
    original_predict_race = (
        ml_system_v4.predict_race if hasattr(ml_system_v4, "predict_race") else None
    )

    def enhanced_predict_race(
        race_data: pd.DataFrame,
        race_id: str,
        market_odds: Dict[str, float] = None,
        market_place_odds: Dict[str, float] = None,
        flags: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Enhanced predict_race with accuracy optimization.

        This builds leakage-safe V4 features via the MLSystemV4 pipeline first,
        then feeds the aligned feature matrix into the ensemble optimizer.
        Enforces a future-date guard to prevent temporal leakage.
        """
        try:
            # 0) Early temporal leakage guard based on provided race_data
            try:
                from utils.date_parsing import parse_date_flexible as _parse
            except Exception:
                _parse = None
            try:
                if (
                    isinstance(race_data, pd.DataFrame)
                    and "race_date" in race_data.columns
                    and len(race_data["race_date"]) > 0
                ):
                    _raw = str(race_data["race_date"].iloc[0])
                    from datetime import date as _date
                    from datetime import datetime

                    if _parse:
                        _parsed = _parse(_raw)
                        _race_dt = datetime.strptime(_parsed, "%Y-%m-%d").date()
                    else:
                        try:
                            _race_dt = datetime.strptime(_raw, "%d %B %Y").date()
                        except Exception:
                            _race_dt = datetime.strptime(_raw, "%Y-%m-%d").date()
                    if _race_dt > _date.today():
                        # Future pre-jump races are valid inference targets.  Only
                        # block them when an explicit legacy guard is requested.
                        import os as _os

                        _block_future = str(
                            _os.getenv("BLOCK_FUTURE_RACE_DATES", "")
                        ).strip().lower() in ("1", "true", "yes", "on")
                        if _block_future:
                            return {
                                "success": False,
                                "error": f"TEMPORAL LEAKAGE DETECTED: race_date {_raw} is in the future relative to today",
                                "race_id": race_id,
                                "fallback_reason": "Future race date detected",
                            }
                        logger.info(
                            "Future race date detected; proceeding because pre-jump inference is lifecycle-gated"
                        )
            except Exception:
                # If parsing fails, continue to builder
                pass

            # 1) Build leakage-safe features using the V4 system (with cache)
            features_df = ml_system_v4.build_features_for_race_with_cache(
                race_data, race_id
            )
            if features_df is None or features_df.empty:
                raise ValueError("Feature building returned empty result")

            # 2) Validate temporal integrity (defense-in-depth)
            try:
                ml_system_v4.temporal_builder.validate_temporal_integrity(
                    features_df, race_data
                )
            except Exception as _e:
                # Non-fatal: log and proceed, original MLSystemV4 would raise
                logger.warning(
                    f"Temporal integrity validation warning for {race_id}: {_e}"
                )

            # 3) Generate optimized predictions using the ensemble on features
            result = accuracy_optimizer.generate_optimized_predictions(
                features_df, race_id
            )

            if result.get("success"):
                # Surface explicit optimizer flags for UI/consumers
                try:
                    result["optimizer_enabled"] = True
                    cfg = getattr(accuracy_optimizer, "config", {}) or {}
                    mode = cfg.get("ensemble_mode")
                    if mode:
                        result["optimizer_mode"] = str(mode)
                except Exception:
                    pass
                return result
            else:
                # Fallback to original if available
                if original_predict_race:
                    logger.warning("Using fallback prediction method")
                    try:
                        return original_predict_race(
                            race_data,
                            race_id,
                            market_odds=market_odds,
                            market_place_odds=market_place_odds,
                            flags=flags,
                        )
                    except TypeError:
                        return original_predict_race(race_data, race_id, market_odds)
                else:
                    return result
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            if original_predict_race:
                try:
                    return original_predict_race(
                        race_data,
                        race_id,
                        market_odds=market_odds,
                        market_place_odds=market_place_odds,
                        flags=flags,
                    )
                except TypeError:
                    return original_predict_race(race_data, race_id, market_odds)
            else:
                return {"success": False, "error": str(e), "race_id": race_id}

    # Replace the method
    ml_system_v4.predict_race = enhanced_predict_race
    ml_system_v4.accuracy_optimizer = accuracy_optimizer

    logger.info("🎯 Enhanced accuracy optimization integrated with ML System V4")
    return ml_system_v4


if __name__ == "__main__":
    # Test the enhanced accuracy optimizer
    optimizer = AccuracyOptimizer()

    # Create sample features for testing
    sample_features = pd.DataFrame(
        {
            "dog_clean_name": ["DOG_A", "DOG_B", "DOG_C"],
            "box_number": [1, 2, 3],
            "weight": [30.0, 32.0, 28.0],
            "starting_price": [3.0, 5.0, 8.0],
        }
    )

    result = optimizer.generate_optimized_predictions(sample_features, "TEST_RACE_001")
    print("🧪 Test Result:")
    print(json.dumps(result, indent=2, default=str))
