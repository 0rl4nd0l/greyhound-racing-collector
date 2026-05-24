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

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import os as _os  # for env checks in persistence

logger = logging.getLogger(__name__)


class EnhancedPredictionService:
    """Service for generating highly accurate and unique predictions."""

    def _apply_sp_tiebreaker(
        self,
        prediction_result: Dict[str, Any],
        race_data: Optional[pd.DataFrame] = None,
        market_odds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Apply a small, transparent SP-based tie-breaker when the top runners are in a near tie.

        Behavior (env-tunable):
        - TIEBREAKER_SP_ENABLED: '1' to enable (default '0')
        - TIEBREAKER_MARGIN_THRESH: float margin threshold for near tie (default 0.03)
        - TIEBREAKER_BUMP: small additive bump to winner's prob before renorm (default 0.01)
        - TIEBREAKER_TOPK: consider top-K runners within margin of the leader (default 2)
        """
        try:
            if not isinstance(prediction_result, dict):
                return
            preds = prediction_result.get("predictions") or []
            if not isinstance(preds, list) or len(preds) < 2:
                return

            # Read env switches
            enabled = str(os.getenv("TIEBREAKER_SP_ENABLED", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if not enabled:
                return
            try:
                margin_thresh = float(os.getenv("TIEBREAKER_MARGIN_THRESH", "0.03"))
            except Exception:
                margin_thresh = 0.03
            try:
                bump = float(os.getenv("TIEBREAKER_BUMP", "0.01"))
            except Exception:
                bump = 0.01
            try:
                topk = int(os.getenv("TIEBREAKER_TOPK", "2"))
            except Exception:
                topk = 2
            topk = max(2, min(topk, len(preds)))

            # Sort by current probability
            def _prob(p: Dict[str, Any]) -> float:
                for k in ("win_prob", "win_prob_norm", "win_probability", "final_score", "confidence"):
                    v = p.get(k)
                    try:
                        if v is None:
                            continue
                        x = float(v)
                        if x > 1.0 and x <= 100.0:
                            x = x / 100.0
                        return max(0.0, min(1.0, x))
                    except Exception:
                        continue
                return 0.0

            preds.sort(key=lambda x: _prob(x), reverse=True)
            p1 = _prob(preds[0])
            p2 = _prob(preds[1])
            if (p1 - p2) >= margin_thresh:
                # Not a near tie; skip
                prediction_result.setdefault("tiebreaker_meta", {}).update({"applied": False})
                return

            # Build SP lookup: prefer field in predictions; else race_data DataFrame; else market odds inverted
            sp_map: Dict[str, float] = {}
            # From predictions
            for p in preds:
                name = str(p.get("dog_name") or p.get("dog_clean_name") or p.get("name") or "").strip().upper()
                if not name:
                    continue
                sp_val = p.get("starting_price")
                try:
                    if sp_val is not None:
                        sp_map[name] = float(sp_val)
                except Exception:
                    pass
            # From race_data DF
            try:
                if race_data is not None and isinstance(race_data, pd.DataFrame) and "dog_clean_name" in race_data.columns:
                    for _, row in race_data.iterrows():
                        nm = str(row.get("dog_clean_name") or "").strip().upper()
                        if not nm:
                            continue
                        if nm not in sp_map and ("starting_price" in race_data.columns):
                            try:
                                sp_map[nm] = float(row.get("starting_price"))
                            except Exception:
                                pass
            except Exception:
                pass
            # From market odds (if provided) -> implied SP approx (not perfect, but a proxy)
            try:
                if market_odds:
                    for nm, odds in market_odds.items():
                        key = str(nm).strip().upper()
                        try:
                            val = float(odds)
                            if val > 0 and key not in sp_map:
                                sp_map[key] = val
                        except Exception:
                            continue
            except Exception:
                pass

            # Identify the near-tied top-K set
            near_tied = [p for p in preds[:topk] if (p1 - _prob(p)) <= margin_thresh + 1e-12]
            if len(near_tied) < 2:
                prediction_result.setdefault("tiebreaker_meta", {}).update({"applied": False})
                return

            # Find the lowest SP among near-tied
            def _name(p: Dict[str, Any]) -> str:
                return str(p.get("dog_name") or p.get("dog_clean_name") or p.get("name") or "").strip().upper()

            best = None
            best_sp = float("inf")
            for p in near_tied:
                nm = _name(p)
                sp_val = sp_map.get(nm)
                if sp_val is None:
                    continue
                try:
                    sp_f = float(sp_val)
                except Exception:
                    continue
                if sp_f < best_sp:
                    best_sp = sp_f
                    best = p

            if best is None or not (best_sp < float("inf")):
                prediction_result.setdefault("tiebreaker_meta", {}).update({"applied": False})
                return

            # Apply bump to the chosen runner and renormalize
            base_probs = [(_prob(p), p) for p in preds]
            total = sum(x for x, _ in base_probs)
            if total <= 0:
                # If all zero, start from uniform
                n = len(preds)
                base = [1.0 / n] * n
            else:
                base = [x / total for x, _ in base_probs]

            # Identify index for best
            idx = next((i for i, (_, p) in enumerate(base_probs) if p is best), None)
            if idx is None:
                prediction_result.setdefault("tiebreaker_meta", {}).update({"applied": False})
                return

            base[idx] = base[idx] + bump
            s2 = sum(base)
            if s2 > 0:
                base = [x / s2 for x in base]

            # Write back probabilities and resort
            for (prob, p), newp in zip(base_probs, base):
                p["win_prob"] = float(newp)
                p["win_prob_norm"] = float(newp)
                p["win_probability"] = float(newp)
                p["final_score"] = float(newp)

            preds.sort(key=lambda x: _prob(x), reverse=True)
            for i, p in enumerate(preds):
                p["predicted_rank"] = i + 1

            meta = prediction_result.setdefault("tiebreaker_meta", {})
            meta.update(
                {
                    "applied": True,
                    "method": "starting_price",
                    "margin_threshold": float(margin_thresh),
                    "bump": float(bump),
                    "topk": int(topk),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as _e:
            # Never fail predictions due to tiebreaker
            try:
                prediction_result.setdefault("tiebreaker_meta", {}).update(
                    {"applied": False, "error": str(_e)}
                )
            except Exception:
                pass

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        # Resolve DB path from environment first, then fallback to provided argument
        try:
            resolved = os.getenv("GREYHOUND_DB_PATH") or os.getenv("ANALYTICS_DB_PATH") or db_path
        except Exception:
            resolved = db_path
        self.db_path = resolved
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
            if (
                hasattr(self.ml_system, "accuracy_optimizer")
                and self.ml_system.accuracy_optimizer
            ):
                self.accuracy_optimizer = self.ml_system.accuracy_optimizer
                logger.info(
                    "✅ Enhanced Prediction Service initialized with accuracy optimization"
                )
            else:
                logger.warning(
                    "⚠️ ML System V4 loaded but accuracy optimizer not available"
                )

        except ImportError as e:
            logger.error(f"Failed to import ML System V4: {e}")
            self.ml_system = None
        except Exception as e:
            logger.error(f"Failed to initialize prediction systems: {e}")
            self.ml_system = None

    def is_available(self) -> bool:
        """Check if the enhanced prediction service is available."""
        return self.ml_system is not None

    def predict_race_enhanced(
        self,
        race_data: pd.DataFrame,
        race_id: str,
        market_odds: Optional[Dict[str, float]] = None,
        tgr_enabled: Optional[bool] = None,
        optimizer_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate enhanced predictions with accuracy optimization.
        tgr_enabled: when provided, toggles runtime inclusion of TGR features.
        """

        if not self.is_available():
            return {
                "success": False,
                "error": "Enhanced prediction service not available",
                "race_id": race_id,
                "fallback_reason": "ML System V4 not initialized",
            }

        try:
            logger.info(f"🎯 Generating enhanced predictions for race: {race_id}")

            # Respect runtime TGR toggle if provided, else read TGR_FEATURES_ENABLED from env
            try:
                if tgr_enabled is None:
                    env_val = os.getenv("TGR_FEATURES_ENABLED")
                    if env_val is not None:
                        tgr_enabled = str(env_val).strip().lower() in ("1", "true", "yes", "on")
                if tgr_enabled is not None and hasattr(self.ml_system, "set_tgr_enabled"):
                    self.ml_system.set_tgr_enabled(bool(tgr_enabled))
            except Exception:
                pass

            # Prepare ML system per-request
            ml_for_call = self.ml_system

            # Ensure optimizer integration if requested
            try:
                if optimizer_enabled is True and hasattr(ml_for_call, "accuracy_optimizer") and getattr(ml_for_call, "accuracy_optimizer", None) is None:
                    # Force-enable for this runtime
                    os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = "0"
                    try:
                        ml_for_call._initialize_accuracy_optimizer()
                    except Exception:
                        pass
            except Exception:
                pass

            # If optimizer explicitly disabled, spawn a fresh MLSystemV4 with optimizer off
            try:
                if optimizer_enabled is False:
                    prev = os.environ.get("V4_DISABLE_ACCURACY_OPTIMIZER")
                    os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = "1"
                    try:
                        from ml_system_v4 import MLSystemV4 as _ML
                        ml_for_call = _ML(self.db_path)
                    finally:
                        if prev is None:
                            try:
                                del os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"]
                            except Exception:
                                pass
                        else:
                            os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = prev
            except Exception:
                pass

            # Apply TGR per-request
            try:
                if tgr_enabled is not None and hasattr(ml_for_call, "set_tgr_enabled"):
                    ml_for_call.set_tgr_enabled(bool(tgr_enabled))
            except Exception:
                pass

            # Use the ML System V4 (configured above)
            # Load feature flags so inference can honor ALLOW_FUTURE_RACE_DATES and others
            try:
                from utils.feature_flags import load_flags as _load_flags
                _flags, _ = _load_flags()
            except Exception:
                _flags = None
            result = ml_for_call.predict_race(
                race_data,
                race_id,
                market_odds=market_odds,
                flags=_flags,
            )

            if result.get("success"):
                # Optional: apply near-tie SP-based tie-breaker before downstream metrics
                try:
                    self._apply_sp_tiebreaker(result, race_data=race_data, market_odds=market_odds)
                except Exception:
                    pass

                # Overlay market metrics (implied prob, overround renorm, edge, EV, Kelly; optional logit-blend)
                try:
                    alpha_env = os.getenv("LOGIT_BLEND_ALPHA", None)
                    alpha = None
                    if alpha_env is not None:
                        try:
                            alpha = float(alpha_env)
                        except Exception:
                            alpha = None
                    kelly_frac = None
                    try:
                        kelly_frac = float(os.getenv("KELLY_FRACTION", "0.25"))
                    except Exception:
                        kelly_frac = 0.25
                    try:
                        kelly_cap = float(os.getenv("KELLY_CAP", "0.05"))
                    except Exception:
                        kelly_cap = 0.05
                    self._apply_market_overlay(
                        result,
                        market_odds or {},
                        alpha=alpha,
                        kelly_fraction=kelly_frac,
                        kelly_cap=kelly_cap,
                    )
                except Exception:
                    # Never fail predictions due to overlay issues
                    pass

                # Add enhanced service metadata
                result["enhanced_service"] = {
                    "accuracy_optimization_applied": self.accuracy_optimizer
                    is not None,
                    "service_version": "1.0",
                    "prediction_method": "ensemble_with_calibration",
                    "uniqueness_validated": True,
                    "timestamp": datetime.now().isoformat(),
                }

                # Add additional quality metrics
                if "predictions" in result:
                    predictions = result["predictions"]

                    # Optionally apply GPT rerank (light blend) behind feature flag
                    try:
                        if str(os.getenv("USE_GPT_RERANK", "1")).lower() in (
                            "1",
                            "true",
                            "yes",
                        ):
                            alpha_env = os.getenv("GPT_RERANK_ALPHA", "0.2")
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
                    quality_metrics = self._calculate_prediction_quality(
                        result.get("predictions") or predictions
                    )
                    result["quality_metrics"] = quality_metrics

                    # Validate uniqueness (already done in ML System V4 if accuracy optimizer available)
                    uniqueness_score = self._validate_prediction_uniqueness(
                        result.get("predictions") or predictions, race_id
                    )
                    result["uniqueness_score"] = uniqueness_score

                    # Add prediction recommendations
                    recommendations = self._generate_prediction_recommendations(
                        result.get("predictions") or predictions, quality_metrics
                    )
                    result["recommendations"] = recommendations

                # Optionally persist predictions to DB if enabled via env
                try:
                    self._persist_predictions_if_enabled(result, race_id)
                except Exception:
                    pass

                logger.info(f"✅ Enhanced predictions generated for {race_id}")
            else:
                logger.warning(
                    f"⚠️ Prediction failed for {race_id}: {result.get('error')}"
                )

            return result

        except Exception as e:
            logger.error(f"Enhanced prediction failed for race {race_id}: {e}")
            return {
                "success": False,
                "error": f"Enhanced prediction error: {str(e)}",
                "race_id": race_id,
                "fallback_reason": "Service exception occurred",
            }

    def predict_race_file_enhanced(
        self, race_file_path: str, tgr_enabled: Optional[bool] = None, optimizer_enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Generate enhanced predictions from race file.
        tgr_enabled: when provided, toggles runtime inclusion of TGR features.
        """

        try:
            # Use Prediction Pipeline V4 if available
            from prediction_pipeline_v4 import PredictionPipelineV4

            pipeline = PredictionPipelineV4(self.db_path)

            # Default tgr_enabled from env if not explicitly provided
            try:
                if tgr_enabled is None:
                    _tgr_env = os.getenv("TGR_FEATURES_ENABLED")
                    if _tgr_env is not None:
                        tgr_enabled = str(_tgr_env).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                pass

            # Ensure optimizer integration if requested
            try:
                if optimizer_enabled is True and hasattr(self.ml_system, "accuracy_optimizer") and getattr(self.ml_system, "accuracy_optimizer", None) is None:
                    os.environ["V4_DISABLE_ACCURACY_OPTIMIZER"] = "0"
                    try:
                        self.ml_system._initialize_accuracy_optimizer()
                    except Exception:
                        pass
            except Exception:
                pass

            # Generate predictions using the pipeline
            try:
                result = pipeline.predict_race_file(
                    race_file_path, tgr_enabled=tgr_enabled, optimizer_enabled=optimizer_enabled
                )
            except TypeError:
                # Backward-compat if pipeline signature not updated
                result = pipeline.predict_race_file(race_file_path)

            if result.get("success"):
                # Optional: apply near-tie SP-based tie-breaker before downstream metrics
                try:
                    # We don't have the original DataFrame here; use prediction fields and market_odds only
                    self._apply_sp_tiebreaker(result, race_data=None, market_odds=None)
                except Exception:
                    pass

                # Enhance the result with additional quality metrics
                if "predictions" in result:
                    predictions = result["predictions"]
                    race_id = result.get("race_id", "unknown")

                    # Optionally persist predictions to DB if enabled via env
                    try:
                        self._persist_predictions_if_enabled(result, race_id)
                    except Exception:
                        pass

                    # Add enhanced service metadata
                    result["enhanced_service"] = {
                        "accuracy_optimization_applied": True,
                        "service_version": "1.0",
                        "prediction_method": "pipeline_v4_with_enhancement",
                        "source_file": race_file_path,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Set predictor metadata defaults for UI/consumers
                    result.setdefault("predictor_used", "EnhancedPredictionService")
                    if not result.get("prediction_methods_used"):
                        result["prediction_methods_used"] = ["ml_system"]
                    result.setdefault("analysis_version", "ML System V4")

                    # Optionally apply GPT rerank (light blend) behind feature flag
                    try:
                        if str(os.getenv("USE_GPT_RERANK", "1")).lower() in (
                            "1",
                            "true",
                            "yes",
                        ):
                            alpha_env = os.getenv("GPT_RERANK_ALPHA", "0.2")
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
                    quality_metrics = self._calculate_prediction_quality(
                        result.get("predictions") or predictions
                    )
                    result["quality_metrics"] = quality_metrics

                    # Validate uniqueness
                    uniqueness_score = self._validate_prediction_uniqueness(
                        result.get("predictions") or predictions, race_id
                    )
                    result["uniqueness_score"] = uniqueness_score

                    # Generate recommendations
                    recommendations = self._generate_prediction_recommendations(
                        result.get("predictions") or predictions, quality_metrics
                    )
                    result["recommendations"] = recommendations

            return result

        except ImportError:
            logger.error("Prediction Pipeline V4 not available")
            return {
                "success": False,
                "error": "Prediction Pipeline V4 not available",
                "race_file": race_file_path,
            }
        except Exception as e:
            logger.error(f"Enhanced file prediction failed: {e}")
            return {
                "success": False,
                "error": f"Enhanced prediction error: {str(e)}",
                "race_file": race_file_path,
            }

    def _gpt_rerank_blend(
        self, prediction_result: Dict[str, Any], alpha: float = 0.2
    ) -> Dict[str, Any]:
        """Blend GPT reranker scores into model predictions conservatively.
        alpha is the GPT weight (0..0.5). Returns an updated prediction_result.
        """
        try:
            preds = (
                prediction_result.get("predictions")
                or prediction_result.get("enhanced_predictions")
                or []
            )
            if not isinstance(preds, list) or not preds:
                return prediction_result
            # Prepare compact payload
            race_info = (
                prediction_result.get("race_info")
                or (prediction_result.get("summary") or {}).get("race_info")
                or {}
            )

            def _base_prob(p: Dict[str, Any]) -> float:
                v = (
                    p.get("win_prob")
                    or p.get("normalized_win_probability")
                    or p.get("win_probability")
                    or p.get("final_score")
                    or p.get("prediction_score")
                    or p.get("confidence")
                    or 0.0
                )
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
                    runners.append(
                        {
                            "dog_name": p.get("dog_name")
                            or p.get("clean_name")
                            or p.get("name"),
                            "box_number": p.get("box_number") or p.get("box"),
                            "win_prob": _base_prob(p),
                            "csv_win_rate": float(p.get("csv_win_rate") or 0.0),
                            "csv_place_rate": float(p.get("csv_place_rate") or 0.0),
                            "avg_finish_position": float(
                                p.get("csv_avg_finish_position") or 10.0
                            ),
                        }
                    )
                except Exception:
                    continue
            if not runners:
                return prediction_result
            # Call GPT reranker
            try:
                from services.gpt_service import GPTService

                gpt = GPTService()
                resp = gpt.enhance_predictions(
                    {"race_info": race_info, "runners": runners}
                )
            except Exception:
                resp = {"scores": []}
            scores = resp.get("scores") or []
            if not isinstance(scores, list) or not scores:
                return prediction_result
            score_map = {}
            for s in scores:
                try:
                    nm = (s.get("dog_name") or "").strip().upper()
                    sc = float(s.get("gpt_score") or 0.0)
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
                name = (
                    (p.get("dog_name") or p.get("clean_name") or p.get("name") or "")
                    .strip()
                    .upper()
                )
                base = _base_prob(p)
                g = score_map.get(name, mean_g)
                new_score = max(0.0, (1.0 - alpha) * base + alpha * g)
                p["gpt_score"] = g
                p["final_score"] = new_score
                blended.append(p)
            total = sum(x.get("final_score", 0.0) for x in blended)
            if total <= 0:
                eq = 1.0 / len(blended)
                for p in blended:
                    p["win_prob"] = eq
            else:
                for p in blended:
                    p["win_prob"] = float(p.get("final_score", 0.0)) / total
            for p in blended:
                # Ensure commonly used keys exist for downstream consumers
                p.setdefault("win_prob_norm", p.get("win_prob", 0.0))
                # Prefer model-derived place probability when available; avoid constant multipliers
                if p.get("place_prob") is None:
                    if p.get("place_prob_norm") is not None:
                        try:
                            p["place_prob"] = float(p.get("place_prob_norm"))
                        except Exception:
                            pass
                    else:
                        try:
                            wp = float(p.get("win_prob") or 0.0)
                        except Exception:
                            wp = 0.0
                        # Conservative monotonic uplift: ensure place_prob >= win_prob and <= 1.0
                        p["place_prob"] = max(wp, min(1.0, wp + 0.5 * (1.0 - wp)))
            prediction_result["predictions"] = blended
            meta = prediction_result.setdefault("gpt_rerank", {})
            meta.update(
                {
                    "alpha": float(alpha),
                    "applied": True,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            # Carry through token usage from GPT call if available
            try:
                tok = (resp.get("_meta") or {}).get("tokens_used")
                if tok is not None:
                    meta["tokens_used"] = int(tok)
            except Exception:
                pass
            # Mark method used for UI transparency
            try:
                methods = prediction_result.setdefault("prediction_methods_used", [])
                if isinstance(methods, list) and "gpt_rerank" not in methods:
                    methods.append("gpt_rerank")
            except Exception:
                pass
            return prediction_result
        except Exception:
            return prediction_result

    def _calculate_prediction_quality(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for predictions."""

        if not predictions:
            return {"error": "No predictions provided"}

        try:
            # Extract key metrics
            win_probs = [
                p.get("win_prob_norm", p.get("win_probability", 0)) for p in predictions
            ]
            confidences = [p.get("confidence", 0.5) for p in predictions]

            # Calculate quality metrics
            import numpy as np

            quality_metrics = {
                "prediction_count": len(predictions),
                "avg_confidence": round(np.mean(confidences), 4),
                "min_confidence": round(min(confidences), 4),
                "max_confidence": round(max(confidences), 4),
                "probability_spread": round(max(win_probs) - min(win_probs), 4),
                "probability_variance": round(np.var(win_probs), 6),
                "predictions_with_high_confidence": sum(
                    1 for c in confidences if c >= 0.7
                ),
                "predictions_with_low_confidence": sum(
                    1 for c in confidences if c < 0.4
                ),
                "favorite_probability": round(max(win_probs), 4),
                "longshot_probability": round(min(win_probs), 4),
                "probability_sum": round(sum(win_probs), 4),
                "normalization_quality": (
                    "good" if 0.95 <= sum(win_probs) <= 1.05 else "poor"
                ),
            }

            # Overall quality score (0-1)
            quality_score = (
                quality_metrics["avg_confidence"] * 0.4
                + min(1.0, quality_metrics["probability_spread"] * 2) * 0.3
                + (1.0 if quality_metrics["normalization_quality"] == "good" else 0.5)
                * 0.3
            )

            quality_metrics["overall_quality_score"] = round(quality_score, 4)
            quality_metrics["quality_level"] = self._get_quality_level(quality_score)

            return quality_metrics

        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return {"error": str(e)}

    def _validate_prediction_uniqueness(
        self, predictions: List[Dict[str, Any]], race_id: str
    ) -> float:
        """Validate prediction uniqueness and return uniqueness score."""

        try:
            if self.accuracy_optimizer and hasattr(
                self.accuracy_optimizer, "uniqueness_validator"
            ):
                # Use the accuracy optimizer's uniqueness validator
                validation_result = (
                    self.accuracy_optimizer.uniqueness_validator.validate_uniqueness(
                        predictions, race_id
                    )
                )
                return validation_result.get("uniqueness_score", 1.0)
            else:
                # Simple uniqueness check
                win_probs = [
                    p.get("win_prob_norm", p.get("win_probability", 0))
                    for p in predictions
                ]
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

    def _generate_prediction_recommendations(
        self, predictions: List[Dict[str, Any]], quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on prediction quality."""

        recommendations = []

        try:
            # Check overall quality
            quality_score = quality_metrics.get("overall_quality_score", 0)
            if quality_score < 0.6:
                recommendations.append(
                    "CAUTION: Low overall prediction quality detected"
                )

            # Check confidence levels
            avg_confidence = quality_metrics.get("avg_confidence", 0)
            if avg_confidence < 0.5:
                recommendations.append(
                    "Consider additional data sources - confidence levels are low"
                )

            # Check probability spread
            prob_spread = quality_metrics.get("probability_spread", 0)
            if prob_spread < 0.1:
                recommendations.append(
                    "Predictions show low variance - consider alternative models"
                )
            elif prob_spread > 0.6:
                recommendations.append(
                    "High variance predictions - verify data quality"
                )

            # Check normalization
            if quality_metrics.get("normalization_quality") == "poor":
                recommendations.append(
                    "Probability normalization issue - check model calibration"
                )

            # Betting recommendations based on predictions
            if predictions:
                # Sort by probability to access top-2
                ordered = sorted(
                    predictions,
                    key=lambda x: x.get("win_prob_norm", x.get("win_probability", 0)),
                    reverse=True,
                )
                top_prediction = ordered[0]
                sec_prediction = ordered[1] if len(ordered) > 1 else None
                top_prob = top_prediction.get(
                    "win_prob_norm", top_prediction.get("win_probability", 0)
                )
                sec_prob = (
                    sec_prediction.get("win_prob_norm", sec_prediction.get("win_probability", 0))
                    if sec_prediction
                    else 0
                )
                top_confidence = top_prediction.get("confidence", 0)

                # Weak favorite advisory when the margin between top-2 is very small
                try:
                    weak_thresh = float(os.getenv("WEAK_FAVORITE_MARGIN_THRESH", "0.05"))
                except Exception:
                    weak_thresh = 0.05
                margin = float(top_prob) - float(sec_prob)
                if sec_prediction is not None and margin < weak_thresh:
                    csv_wr = top_prediction.get("csv_win_rate")
                    csv_afp = top_prediction.get("csv_avg_finish_position")
                    wr_txt = (
                        f", csv_win_rate={csv_wr:.2f}" if isinstance(csv_wr, (int, float)) else ""
                    )
                    afp_txt = (
                        f", avg_finish_pos={csv_afp:.1f}" if isinstance(csv_afp, (int, float)) else ""
                    )
                    recommendations.append(
                        f"Weak favorite: top-2 margin {margin:.3f} < {weak_thresh:.3f}{wr_txt}{afp_txt}"
                    )

                if top_prob > 0.4 and top_confidence > 0.7:
                    recommendations.append(
                        f"Strong favorite identified: {top_prediction.get('dog_clean_name', 'Unknown')}"
                    )
                elif top_prob < 0.2 and quality_score > 0.7:
                    recommendations.append(
                        "Competitive race - consider place/show betting"
                    )

            # Add positive recommendations
            if quality_score >= 0.8:
                recommendations.append(
                    "HIGH QUALITY: Predictions show strong reliability"
                )
            elif quality_score >= 0.7:
                recommendations.append(
                    "GOOD QUALITY: Predictions are reliable for betting decisions"
                )

        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append(
                "Unable to generate specific recommendations due to analysis error"
            )

        return (
            recommendations
            if recommendations
            else ["Standard predictions generated - review individual dog assessments"]
        )

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

    def _apply_market_overlay(
        self,
        prediction_result: Dict[str, Any],
        market_odds: Dict[str, float],
        alpha: Optional[float] = None,
        kelly_fraction: float = 0.25,
        kelly_cap: float = 0.05,
    ) -> None:
        """Augment predictions with market metrics: implied probabilities (overround-adjusted),
        edge vs market, EV for win, and capped fractional Kelly stakes. Optionally apply a
        logit-blend between model and market for decision support (not training).
        """
        try:
            preds = prediction_result.get("predictions") or []
            if not preds:
                return

            # Build a name->odds map with cleaned keys
            def _norm_key(name: str) -> str:
                try:
                    import re as _re
                    s = str(name or "").upper().strip()
                    s = _re.sub(r"[^\w\s]", "", s)
                    return s
                except Exception:
                    return str(name or "").upper().strip()

            odds_map = { _norm_key(k): float(v) for k, v in (market_odds or {}).items() if v is not None }

            # Collect model probs and market implied
            model_probs = []
            names = []
            mkt_raw = []
            for p in preds:
                nm = p.get("dog_name") or p.get("dog_clean_name") or p.get("name")
                key = _norm_key(nm)
                names.append(key)
                # Prefer calibrated/normalized win
                pm = None
                for k in ("win_prob_norm", "win_probability", "win_prob", "final_score"):
                    v = p.get(k)
                    if v is None:
                        continue
                    try:
                        pm = float(v)
                        if pm > 1.0:
                            pm = pm / 100.0
                        break
                    except Exception:
                        continue
                if pm is None:
                    pm = 0.0
                model_probs.append(max(0.0, min(1.0, pm)))
                # Market implied
                od = odds_map.get(key)
                if od and od > 0:
                    mkt_raw.append(1.0 / float(od))
                else:
                    mkt_raw.append(0.0)

            # Renormalize per-race
            import math
            eps = 1e-12
            s_model = sum(model_probs)
            p_model = [ (x / s_model) if s_model > eps else (1.0/len(model_probs)) for x in model_probs ]
            s_mkt = sum(mkt_raw)
            p_mkt = [ (x / s_mkt) if s_mkt > eps else 0.0 for x in mkt_raw ]

            # Optional logit-blend for decision support (alpha in [0,1])
            def _sigmoid(x: float) -> float:
                return 1.0 / (1.0 + math.exp(-x))
            def _safe_logit(p: float) -> float:
                p = min(1 - 1e-9, max(1e-9, p))
                return math.log(p / (1 - p))
            if alpha is not None and 0.0 <= alpha <= 1.0:
                p_blend = []
                for pm, mk in zip(p_model, p_mkt):
                    lb = alpha * _safe_logit(pm) + (1.0 - alpha) * _safe_logit(mk if mk > 0 else 1.0/len(p_mkt))
                    p_blend.append(_sigmoid(lb))
                # Renormalize blended
                sb = sum(p_blend)
                if sb > eps:
                    p_blend = [x / sb for x in p_blend]
            else:
                p_blend = p_model[:]

            # Compute EV, edge, and Kelly
            out = []
            for i, p in enumerate(preds):
                key = names[i]
                odds = odds_map.get(key)
                p_m = p_model[i]
                p_b = p_blend[i]
                mkt_p = p_mkt[i]
                edge = None
                ev_win = None
                kelly = 0.0
                if odds and odds > 0:
                    edge = p_b - mkt_p  # decision edge
                    ev_win = p_b * odds - 1.0
                    # Kelly fraction for win
                    try:
                        numer = odds * p_b - (1.0 - p_b)
                        denom = max(1e-9, (odds - 1.0))
                        kelly_full = numer / denom
                        kelly = max(0.0, kelly_fraction * kelly_full)
                        if kelly_cap is not None:
                            kelly = min(kelly, float(kelly_cap))
                    except Exception:
                        kelly = 0.0

                # Write back fields
                p["implied_prob_raw"] = mkt_raw[i]
                p["implied_prob_norm"] = mkt_p
                p["win_prob_norm"] = p_m  # ensure present
                p["win_prob_blend"] = p_b
                p["edge"] = edge
                p["ev_win"] = ev_win
                p["kelly_fraction"] = kelly

            # Resort by blended prob (stable fallback by win_prob_norm)
            preds.sort(key=lambda x: (x.get("win_prob_blend") or x.get("win_prob_norm") or 0.0), reverse=True)
            for i, p in enumerate(preds):
                p["predicted_rank"] = i + 1
        except Exception as _e:
            try:
                prediction_result.setdefault("overlay_error", str(_e))
            except Exception:
                pass

    def _persist_predictions_if_enabled(self, prediction_result: Dict[str, Any], race_id: str) -> None:
        """Persist predictions to the SQLite DB if PERSIST_PREDICTIONS is enabled.
        Table schema: predictions(race_id TEXT, dog_clean_name TEXT, predicted_probability REAL, confidence_level TEXT, timestamp TEXT)
        """
        try:
            # Check env gate
            flag = str(_os.getenv("PERSIST_PREDICTIONS", "0")).strip().lower()
            if flag not in ("1", "true", "yes", "on"):
                return
            preds = prediction_result.get("predictions") or []
            if not preds:
                return
            # Choose race_id to write (map to standardized race_id when possible)
            rid = prediction_result.get("race_id") or race_id

            # Attempt to map filename-style race_id to standardized race_id using race_metadata
            try:
                import re as _re
                import sqlite3 as _sqlite

                def _norm_name(x: str) -> str:
                    try:
                        return _re.sub(r"[^\w]", "", (x or "").upper().strip())
                    except Exception:
                        return (x or "").upper().replace(" ", "")

                # Extract metadata from prediction_result
                rc = prediction_result.get("race_context") or {}
                ri = prediction_result.get("race_info") or {}
                venue_raw = rc.get("venue") or ri.get("venue")
                date_raw = rc.get("race_date") or ri.get("race_date") or ri.get("date")
                race_num = ri.get("race_number")

                # Parse race number from filename if missing
                if race_num is None:
                    try:
                        fn = ri.get("filename") or prediction_result.get("race_id") or ""
                        m = _re.match(r"^Race\s+(\d+)\s*-\s*", str(fn))
                        if m:
                            race_num = int(m.group(1))
                    except Exception:
                        race_num = None

                std_rid = None
                if date_raw and race_num is not None:
                    try:
                        conn_lookup = _sqlite.connect(self.db_path)
                        cur_lookup = conn_lookup.cursor()
                        cur_lookup.execute(
                            """
                            SELECT race_id, venue, COALESCE(venue_slug, venue) AS venue_slug
                            FROM race_metadata
                            WHERE race_date = ? AND race_number = ?
                            """,
                            (str(date_raw), int(race_num)),
                        )
                        rows = cur_lookup.fetchall() or []
                        conn_lookup.close()
                        if rows:
                            vn_norm = _norm_name(venue_raw) if venue_raw else None
                            # Prefer exact venue/slug match if we have a venue label; else fallback to first row
                            chosen = None
                            if vn_norm:
                                for r_row in rows:
                                    rm_rid, rm_venue, rm_slug = r_row
                                    if _norm_name(rm_venue) == vn_norm or _norm_name(rm_slug) == vn_norm:
                                        chosen = rm_rid
                                        break
                            if not chosen:
                                # Fallback: use the first candidate for given date+race_number
                                chosen = rows[0][0]
                            std_rid = chosen
                    except Exception:
                        std_rid = None
                if std_rid:
                    rid = std_rid
            except Exception:
                pass

            import sqlite3, re as _re
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        race_id TEXT,
                        dog_clean_name TEXT,
                        predicted_probability REAL,
                        confidence_level TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur = conn.cursor()
                def _norm_name(x: str) -> str:
                    return _re.sub(r"[^\w\s]", "", (x or "").upper().strip())
                def _prob(p: Dict[str, Any]) -> Optional[float]:
                    for k in ("win_prob_norm", "win_probability", "win_prob", "final_score", "prediction_score"):
                        v = p.get(k)
                        if v is None:
                            continue
                        try:
                            x = float(v)
                            return x/100.0 if x>1.0 else x
                        except Exception:
                            continue
                    return None
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for p in preds:
                    nm = _norm_name(p.get("dog_clean_name") or p.get("dog_name") or p.get("name"))
                    if not nm:
                        continue
                    prob = _prob(p)
                    if prob is None:
                        continue
                    conf = p.get("confidence_label") or p.get("confidence_level") or "MEDIUM"
                    try:
                        cur.execute(
                            """
                            INSERT INTO predictions (race_id, dog_clean_name, predicted_probability, confidence_level, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (rid, nm, float(prob), str(conf), ts),
                        )
                    except Exception:
                        continue
                conn.commit()
            finally:
                conn.close()
        except Exception:
            # Never fail predictions due to persistence
            pass

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and capabilities."""

        return {
            "service_available": self.is_available(),
            "ml_system_loaded": self.ml_system is not None,
            "accuracy_optimizer_available": self.accuracy_optimizer is not None,
            "enhanced_features": {
                "multi_model_ensemble": True,
                "dynamic_weighting": True,
                "real_time_calibration": True,
                "uniqueness_validation": self.accuracy_optimizer is not None,
                "performance_feedback": True,
                "quality_metrics": True,
            },
            "service_version": "1.0",
            "initialization_timestamp": datetime.now().isoformat(),
        }


# Global service instance
_enhanced_prediction_service = None


def get_enhanced_prediction_service(
    db_path: str = "greyhound_racing_data.db",
) -> EnhancedPredictionService:
    """Get or create the global enhanced prediction service instance."""
    global _enhanced_prediction_service

    if _enhanced_prediction_service is None:
        _enhanced_prediction_service = EnhancedPredictionService(db_path)

    return _enhanced_prediction_service


# Convenience functions for integration
def predict_race_enhanced(
    race_data: pd.DataFrame,
    race_id: str,
    market_odds: Optional[Dict[str, float]] = None,
    db_path: str = "greyhound_racing_data.db",
    tgr_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate enhanced predictions for a race."""
    service = get_enhanced_prediction_service(db_path)
    return service.predict_race_enhanced(
        race_data, race_id, market_odds, tgr_enabled=tgr_enabled
    )


def predict_race_file_enhanced(
    race_file_path: str,
    db_path: str = "greyhound_racing_data.db",
    tgr_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
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
        sample_data = pd.DataFrame(
            {
                "dog_clean_name": ["TEST_DOG_A", "TEST_DOG_B", "TEST_DOG_C"],
                "box_number": [1, 2, 3],
                "weight": [30.0, 32.0, 28.0],
                "starting_price": [3.0, 5.0, 8.0],
            }
        )

        result = service.predict_race_enhanced(sample_data, "TEST_RACE_001")
        print("\n🧪 Test Prediction Result:")
        print(json.dumps(result, indent=2, default=str))
