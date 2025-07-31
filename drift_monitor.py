import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr

import logging

# Try to import evidently, fall back to manual PSI/KS calculation if not available
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️ Evidently not available, using manual drift detection")

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor data drift between training and incoming prediction data."""
    
    def __init__(self, reference_data: pd.DataFrame, correlation_threshold: float = 0.3):
        self.reference_data = reference_data.copy()
        self.correlation_threshold = correlation_threshold
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Initialize evidently if available
        if EVIDENTLY_AVAILABLE:
            self.drift_report = Report(metrics=[DataDriftPreset()])
            self.column_mapping = ColumnMapping()
        
        # Calculate baseline correlations
        self._calculate_baseline_correlations()
        
        logger.info(f"DriftMonitor initialized with {len(self.reference_data)} reference samples")
    
    def _calculate_baseline_correlations(self):
        """Calculate baseline Pearson and Spearman correlations."""
        try:
            # Only use numeric columns for correlation
            numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
            self.baseline_pearson = self.reference_data[numeric_cols].corr(method='pearson')
            self.baseline_spearman = self.reference_data[numeric_cols].corr(method='spearman')
            
            logger.info(f"Calculated baseline correlations for {len(numeric_cols)} numeric features")
        except Exception as e:
            logger.error(f"Error calculating baseline correlations: {e}")
            self.baseline_pearson = pd.DataFrame()
            self.baseline_spearman = pd.DataFrame()
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI) manually."""
        try:
            # Handle non-numeric data by converting to string and using value counts
            if not pd.api.types.is_numeric_dtype(expected):
                expected_counts = expected.value_counts(normalize=True, sort=False)
                actual_counts = actual.value_counts(normalize=True, sort=False)
                
                # Align the series
                all_values = set(expected_counts.index) | set(actual_counts.index)
                expected_aligned = expected_counts.reindex(all_values, fill_value=0.001)
                actual_aligned = actual_counts.reindex(all_values, fill_value=0.001)
            else:
                # For numeric data, create buckets
                try:
                    expected_counts, bin_edges = np.histogram(expected.dropna(), bins=buckets, density=True)
                    actual_counts, _ = np.histogram(actual.dropna(), bins=bin_edges, density=True)
                    
                    # Normalize to probabilities
                    expected_counts = expected_counts / expected_counts.sum()
                    actual_counts = actual_counts / actual_counts.sum()
                    
                    # Add small epsilon to avoid log(0)
                    expected_counts = np.maximum(expected_counts, 0.001)
                    actual_counts = np.maximum(actual_counts, 0.001)
                    
                    expected_aligned = expected_counts
                    actual_aligned = actual_counts
                except Exception:
                    # Fallback to simple approach
                    return 0.0
            
            # Calculate PSI
            psi = np.sum((actual_aligned - expected_aligned) * np.log(actual_aligned / expected_aligned))
            return float(psi)
            
        except Exception as e:
            logger.debug(f"Error calculating PSI: {e}")
            return 0.0
    
    def check_for_drift(self, current_data: pd.DataFrame) -> Dict:
        """Check for data drift and log results."""
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(self.reference_data),
            "current_size": len(current_data),
            "drift_detected": False,
            "correlation_alerts": [],
            "feature_drift": {},
            "summary": {"high_drift_features": [], "moderate_drift_features": []}
        }
        
        try:
            # Use evidently if available
            if EVIDENTLY_AVAILABLE:
                drift_results.update(self._evidently_drift_check(current_data))
            else:
                drift_results.update(self._manual_drift_check(current_data))
            
            # Check correlations
            correlation_alerts = self._check_correlations(current_data)
            drift_results["correlation_alerts"] = correlation_alerts
            
            if correlation_alerts:
                drift_results["drift_detected"] = True
            
            # Log results
            self._log_drift_results(drift_results)
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            drift_results["error"] = str(e)
        
        return drift_results
    
    def _evidently_drift_check(self, current_data: pd.DataFrame) -> Dict:
        """Use evidently for drift detection."""
        try:
            # Align columns between reference and current data
            common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
            if not common_cols:
                return {"error": "No common columns between reference and current data"}
            
            ref_aligned = self.reference_data[common_cols]
            curr_aligned = current_data[common_cols]
            
            # Run drift report
            self.drift_report.run(
                reference_data=ref_aligned, 
                current_data=curr_aligned, 
                column_mapping=self.column_mapping
            )
            
            # Extract results
            return {
                "evidently_results": self.drift_report.as_dict(),
                "method": "evidently"
            }
            
        except Exception as e:
            logger.warning(f"Evidently drift check failed: {e}")
            return self._manual_drift_check(current_data)
    
    def _manual_drift_check(self, current_data: pd.DataFrame) -> Dict:
        """Manual drift detection using PSI and KS test."""
        feature_drift = {}
        high_drift_features = []
        moderate_drift_features = []
        
        # Get common columns
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        
        for col in common_cols:
            try:
                ref_col = self.reference_data[col].dropna()
                curr_col = current_data[col].dropna()
                
                if len(ref_col) == 0 or len(curr_col) == 0:
                    continue
                
                # Calculate PSI
                psi = self._calculate_psi(ref_col, curr_col)
                
                # Calculate KS statistic for numeric columns
                ks_stat = 0.0
                ks_pvalue = 1.0
                if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(curr_col):
                    try:
                        ks_stat, ks_pvalue = ks_2samp(ref_col, curr_col)
                    except Exception:
                        pass
                
                feature_drift[col] = {
                    "psi": psi,
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue)
                }
                
                # Classify drift level
                if psi > 0.25 or ks_pvalue < 0.01:
                    high_drift_features.append(col)
                elif psi > 0.1 or ks_pvalue < 0.05:
                    moderate_drift_features.append(col)
                    
            except Exception as e:
                logger.debug(f"Error processing column {col}: {e}")
        
        return {
            "method": "manual",
            "feature_drift": feature_drift,
            "summary": {
                "high_drift_features": high_drift_features,
                "moderate_drift_features": moderate_drift_features
            }
        }
    
    def _check_correlations(self, current_data: pd.DataFrame) -> list:
        """Check for significant correlation changes."""
        alerts = []
        
        try:
            # Only use numeric columns
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            common_cols = list(set(self.baseline_pearson.columns) & set(numeric_cols))
            
            if len(common_cols) < 2:
                return alerts
            
            # Calculate current correlations
            current_pearson = current_data[common_cols].corr(method='pearson')
            current_spearman = current_data[common_cols].corr(method='spearman')
            
            # Check Pearson correlation changes
            pearson_diff = current_pearson - self.baseline_pearson.loc[common_cols, common_cols]
            
            for col1 in common_cols:
                for col2 in common_cols:
                    if col1 != col2:
                        pearson_change = abs(pearson_diff.loc[col1, col2])
                        if pearson_change > self.correlation_threshold and not np.isnan(pearson_change):
                            alert = {
                                "type": "pearson_correlation_change",
                                "feature_1": col1,
                                "feature_2": col2,
                                "change": float(pearson_change),
                                "baseline_corr": float(self.baseline_pearson.loc[col1, col2]),
                                "current_corr": float(current_pearson.loc[col1, col2])
                            }
                            alerts.append(alert)
                            
                            logger.warning(
                                f"Significant Pearson correlation change: {col1} vs {col2}: "
                                f"{pearson_change:.3f} (baseline: {self.baseline_pearson.loc[col1, col2]:.3f}, "
                                f"current: {current_pearson.loc[col1, col2]:.3f})"
                            )
            
            # Check Spearman correlation changes  
            spearman_diff = current_spearman - self.baseline_spearman.loc[common_cols, common_cols]
            
            for col1 in common_cols:
                for col2 in common_cols:
                    if col1 != col2:
                        spearman_change = abs(spearman_diff.loc[col1, col2])
                        if spearman_change > self.correlation_threshold and not np.isnan(spearman_change):
                            alert = {
                                "type": "spearman_correlation_change",
                                "feature_1": col1,
                                "feature_2": col2,
                                "change": float(spearman_change),
                                "baseline_corr": float(self.baseline_spearman.loc[col1, col2]),
                                "current_corr": float(current_spearman.loc[col1, col2])
                            }
                            alerts.append(alert)
                            
                            logger.warning(
                                f"Significant Spearman correlation change: {col1} vs {col2}: "
                                f"{spearman_change:.3f} (baseline: {self.baseline_spearman.loc[col1, col2]:.3f}, "
                                f"current: {current_spearman.loc[col1, col2]:.3f})"
                            )
        
        except Exception as e:
            logger.error(f"Error checking correlations: {e}")
        
        return alerts
    
    def _log_drift_results(self, drift_results: Dict):
        """Log drift results to JSON file."""
        try:
            log_file_name = f"logs/drift_report_{datetime.now().date()}.json"
            
            # If file exists, load and append
            if os.path.exists(log_file_name):
                try:
                    with open(log_file_name, 'r') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except:
                    existing_data = []
            else:
                existing_data = []
            
            existing_data.append(drift_results)
            
            with open(log_file_name, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
            
            logger.info(f"Drift report logged to {log_file_name}")
            
        except Exception as e:
            logger.error(f"Error logging drift results: {e}")
