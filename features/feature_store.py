import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime
import hashlib

class FeatureStore:
    """Feature store for managing feature versions and persistence."""

    def __init__(self, path='feature_store.parquet'):
        self.path = path
        self.feature_version = "3.0.0"

    def persist(self, features):
        """Persist features to parquet file with metadata for version and hash."""
        metadata = {'feature_version': self.feature_version,
                    'data_hash': self._calculate_hash(features)}
        df = pd.DataFrame(features)
        df.attrs.update(metadata)
        df.to_parquet(self.path, index=False)

    def load(self):
        """Load features from the parquet file."""
        return pd.read_parquet(self.path)

    def _calculate_hash(self, features):
        """Calculate a hash for the given features to track changes and avoid duplicates."""
        feature_str = pd.util.hash_pandas_object(pd.DataFrame(features), index=True).sum()
        return hashlib.md5(str(feature_str).encode()).hexdigest()

    def check_drift(self, current_features, baseline_features):
        """Check for distribution drift between current and baseline features using KS test."""
        drifts = {}
        for col in current_features.columns:
            stat, p_value = ks_2samp(current_features[col], baseline_features[col])
            drift_detected = p_value < 0.05  # Using 5% significance level
            drifts[col] = {
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': drift_detected
            }
        return drifts

