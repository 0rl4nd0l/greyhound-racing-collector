import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Optional heavy deps: make safe for constrained test envs without numpy/pandas/scipy
try:  # pragma: no cover - environment dependent
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    np = None

try:  # pragma: no cover - environment dependent
    import pandas as pd  # noqa: F401
except Exception:  # pragma: no cover
    pd = None

# If pandas is available but parquet engines are not, provide a safe read_parquet shim
if pd is not None:  # pragma: no cover - environment dependent
    try:
        _orig_read_parquet = pd.read_parquet

        def _safe_read_parquet(*args, **kwargs):
            try:
                return _orig_read_parquet(*args, **kwargs)
            except ImportError:
                # Signal to callers that parquet is effectively unavailable
                raise FileNotFoundError("Parquet engine unavailable")

        pd.read_parquet = _safe_read_parquet
    except Exception:
        # If anything goes wrong, leave pandas as-is
        pass

try:  # pragma: no cover - environment dependent
    from scipy.stats import ks_2samp  # type: ignore
except Exception:  # pragma: no cover
    ks_2samp = None  # Disable drift detection without SciPy


class FeatureStore:
    """Feature store for managing feature versions and persistence.

    Designed to degrade gracefully when heavy scientific libraries are unavailable
    (e.g., during tests on systems without numpy/pandas/scipy). In such cases,
    persistence/drift-check features are no-ops or raise clear ImportErrors when
    functionality truly requires pandas.
    """

    def __init__(self, path="feature_store.parquet"):
        self.path = path
        self.feature_version = "3.0.0"

    def persist(self, features):
        """Persist features to parquet file with metadata for version and hash."""
        if pd is None:
            # Graceful no-op persist to a JSON file so tests don't fail
            import json

            meta = {
                "feature_version": self.feature_version,
                "data_hash": self._simple_hash(features),
                "storage": "json_fallback",
            }
            payload = {"metadata": meta, "data": features}
            json_path = os.path.splitext(self.path)[0] + ".json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return
        metadata = {
            "feature_version": self.feature_version,
            "data_hash": self._calculate_hash(features),
        }
        df = pd.DataFrame(features)
        try:
            df.attrs.update(metadata)
        except Exception:
            # attrs not critical for persistence
            pass
        try:
            df.to_parquet(self.path, index=False)
        except Exception as e:
            # Fallback when parquet engine is unavailable
            try:
                csv_path = os.path.splitext(self.path)[0] + ".csv"
                df.to_csv(csv_path, index=False)
            except Exception:
                # Last resort JSON fallback
                import json

                payload = {"metadata": metadata, "data": df.to_dict(orient="records")}
                json_path = os.path.splitext(self.path)[0] + ".json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

    def load(self):
        """Load features from the parquet file or JSON fallback if pandas unavailable."""
        if pd is None:
            import json

            json_path = os.path.splitext(self.path)[0] + ".json"
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload.get("data", [])
            # If pandas is unavailable and no JSON fallback exists
            raise ImportError("pandas not available to load feature store")
        try:
            return pd.read_parquet(self.path)
        except Exception:
            # Try CSV fallback
            csv_path = os.path.splitext(self.path)[0] + ".csv"
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            # Try JSON fallback
            import json

            json_path = os.path.splitext(self.path)[0] + ".json"
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                # Return DataFrame if pandas available, else raw
                data = payload.get("data", [])
                return pd.DataFrame(data)
            raise

    def _simple_hash(self, features):
        return hashlib.md5(str(features).encode()).hexdigest()

    def _calculate_hash(self, features):
        """Calculate a hash for the given features to track changes and avoid duplicates."""
        if pd is None:
            return self._simple_hash(features)
        feature_str = pd.util.hash_pandas_object(
            pd.DataFrame(features), index=True
        ).sum()
        return hashlib.md5(str(feature_str).encode()).hexdigest()

    def check_drift(self, current_features, baseline_features):
        """Check for distribution drift between current and baseline features using KS test.

        If dependencies are missing, return an empty report to avoid breaking callers.
        """
        if pd is None or np is None or ks_2samp is None:
            logging.warning(
                "DRIFT_DETECTION_DISABLED missing dependencies (pandas/numpy/scipy)"
            )
            return {}
        drifts = {}
        for col in current_features.columns:
            try:
                stat, p_value = ks_2samp(current_features[col], baseline_features[col])
                drift_detected = p_value < 0.05  # Using 5% significance level
                drifts[col] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": bool(drift_detected),
                }
            except Exception:
                # Skip columns that can't be compared
                continue
        return drifts

    def load_v4_model_contract(self):
        """Load the canonical V4 model feature contract.

        Preference order for expected features:
        1) numerical_columns + categorical_columns (keeps saved order)
        2) features (legacy flat list fallback)
        """
        # Candidate paths (repo-root and module-relative)
        candidates = [
            Path("docs/model_contracts/v4_feature_contract.json"),
            Path(__file__).parent.parent
            / "docs/model_contracts/v4_feature_contract.json",
        ]
        contract_path = None
        for p in candidates:
            try:
                if p.exists():
                    contract_path = p
                    break
            except Exception:
                continue

        if contract_path is None:
            raise FileNotFoundError(
                "V4 model contract not found at docs/model_contracts/v4_feature_contract.json. "
                "Run 'python3 scripts/verify_feature_contract.py --refresh' to generate it."
            )

        with open(contract_path, "r", encoding="utf-8") as f:
            try:
                contract = json.load(f) or {}
            except Exception as e:
                raise ValueError(
                    f"Failed to read V4 contract JSON at {contract_path}: {e}"
                )

        # Prefer explicit numerical/categorical columns; fall back to legacy flat list
        expected_features = []
        try:
            nums = contract.get("numerical_columns") or []
            cats = contract.get("categorical_columns") or []
            if nums or cats:
                expected_features = list(nums) + list(cats)
        except Exception:
            expected_features = []

        if not expected_features:
            # Legacy fallback: flat ordered feature list
            flist = contract.get("features")
            if isinstance(flist, list) and flist:
                expected_features = list(flist)

        if not expected_features:
            raise ValueError(
                "V4 contract present but contains no 'numerical_columns'/'categorical_columns' or 'features' list"
            )

        return expected_features

    def enforce_v4_contract(self, features_df, log_missing=True):
        """Enforce V4 model contract by reindexing DataFrame to match expected features.

        Args:
            features_df: DataFrame with computed features
            log_missing: Whether to log missing columns that are backfilled with NaN

        Returns:
            DataFrame with columns reordered and missing columns filled with NaN
        """
        if pd is None:
            raise ImportError("pandas required for contract enforcement")

        try:
            expected_features = self.load_v4_model_contract()
        except FileNotFoundError as e:
            if log_missing:
                logging.warning(f"Could not load V4 contract: {e}")
            return features_df

        # Get current columns
        current_columns = set(features_df.columns)
        expected_columns = set(expected_features)

        # Find missing columns
        missing_columns = expected_columns - current_columns
        extra_columns = current_columns - expected_columns

        if missing_columns and log_missing:
            logging.warning(
                f"Missing {len(missing_columns)} features for V4 model: {sorted(missing_columns)[:5]}..."
            )

        if extra_columns and log_missing:
            logging.debug(
                f"Extra {len(extra_columns)} features not used by V4 model: {sorted(extra_columns)[:5]}..."
            )

        # Reindex to match contract (adds missing columns with NaN)
        contract_aligned_df = features_df.reindex(columns=expected_features)

        # Apply dtype casting for consistency
        contract_aligned_df = self._cast_feature_dtypes(contract_aligned_df)

        # Log successful alignment
        if log_missing:
            missing_count = contract_aligned_df.isnull().sum().sum()
            logging.info(
                f"✅ Aligned {len(contract_aligned_df.columns)} features to V4 contract "
                f"({missing_count} NaN values introduced for missing features)"
            )

        return contract_aligned_df

    def _cast_feature_dtypes(self, df):
        """Cast feature columns to appropriate dtypes for model consistency.

        Important: avoid pandas nullable integer dtypes (Int64) because they use
        pandas.NA, which can cause issues when downstream scikit-learn transformers
        materialize numpy arrays. We prefer float64 for all numeric features and
        replace missing values for count-like fields with 0.0.
        """
        if pd is None or df.empty:
            return df

        # Define expected dtypes for different feature categories
        count_like_features = {
            "box_number",
            "field_size",
            "venue_experience",
            "grade_experience",
        }

        float_features = {
            "distance",
            "weight",
            "temperature",
            "humidity",
            "wind_speed",
            "historical_avg_position",
            "historical_best_position",
            "historical_win_rate",
            "historical_place_rate",
            "historical_form_trend",
            "historical_avg_time",
            "historical_best_time",
            "historical_time_consistency",
            "target_distance",
            "venue_specific_avg_position",
            "venue_specific_win_rate",
            "venue_best_position",
            "grade_specific_avg_position",
            "grade_specific_win_rate",
            "days_since_last_race",
            "race_frequency",
            "best_distance_avg_position",
            "best_distance_win_rate",
        }

        bool_features = {"distance_adjusted_time"}

        # Apply dtype casting
        for col in df.columns:
            try:
                if col in count_like_features:
                    # Cast to float64 and fill missing with 0.0 to avoid pandas.NA
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce")
                        .astype("float64")
                        .fillna(0.0)
                    )
                elif col in float_features:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                elif col in bool_features:
                    df[col] = df[col].astype("bool")
                # String features (venue, grade, etc.) are left as-is
            except Exception as e:
                logging.debug(f"Could not cast dtype for column {col}: {e}")

        # Final safety: ensure no pandas.NA remain in numeric columns
        try:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].astype("float64")
        except Exception as e:
            logging.debug(f"Numeric dtype normalization failed: {e}")

        return df

    def validate_v4_features(self, features_df):
        """Validate that features DataFrame is ready for V4 model prediction.

        Returns:
            dict with validation results
        """
        if pd is None:
            return {"valid": False, "error": "pandas not available"}

        try:
            expected_features = self.load_v4_model_contract()
        except FileNotFoundError:
            return {"valid": False, "error": "V4 contract file not found"}

        current_columns = set(features_df.columns)
        expected_columns = set(expected_features)

        missing_columns = expected_columns - current_columns

        if missing_columns:
            return {
                "valid": False,
                "error": f"Missing required columns: {sorted(missing_columns)}",
                "missing_columns": list(missing_columns),
                "current_columns": list(current_columns),
            }

        # Check for NaN values in critical features
        critical_features = {"box_number", "weight", "distance", "venue", "grade"}

        nan_critical = []
        for col in critical_features:
            if col in features_df.columns and features_df[col].isnull().any():
                nan_critical.append(col)

        return {
            "valid": True,
            "feature_count": len(features_df.columns),
            "missing_columns": [],
            "nan_critical_features": nan_critical,
            "total_nan_values": features_df.isnull().sum().sum(),
        }
