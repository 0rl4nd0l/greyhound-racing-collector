#!/usr/bin/env python3
"""
Baseline Statistics Management System
====================================

This system manages baseline statistics for model monitoring and drift detection.
It creates versioned baseline statistics from reference data and provides
traceability through model SHA and date versioning.

Features:
- Load existing baseline statistics or create from reference data
- Version statistics with model SHA and date for traceability
- Calculate comprehensive statistics (mean, std, quantiles) for monitored features
- Automatic fallback to reference data when baseline stats are missing
- Integration with model registry system

Author: AI Assistant
Date: August 3, 2025
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

logger = logging.getLogger(__name__)


class BaselineStatsManager:
    """
    Manages baseline statistics for model monitoring and drift detection
    """
    
    def __init__(self, 
                 baseline_dir: str = "./baseline_stats",
                 reference_db_path: str = "./databases/race_data.db"):
        self.baseline_dir = Path(baseline_dir)
        self.reference_db_path = reference_db_path
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Get current model version info
        self.git_sha = self._get_git_sha()
        self.date = datetime.now().strftime('%Y%m%d')
        self.model_version = f"{self.git_sha}_{self.date}"
        
        logger.info(f"📊 Baseline Stats Manager initialized for version: {self.model_version}")

    def _get_git_sha(self) -> str:
        """Get current git SHA for versioning"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not get git SHA, using timestamp")
            return datetime.now().strftime('%H%M%S')

    def get_monitored_features(self) -> List[str]:
        """
        Define the features to monitor for baseline statistics
        These should align with your model's input features
        """
        return [
            "weight",
            "box_number",
            "distance"
        ]

    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data from the database"""
        try:
            conn = sqlite3.connect(self.reference_db_path)
            
            # Comprehensive query to get all relevant features
            query = """
            SELECT 
                dp.weight,
                dp.race_time,
                dp.finish_position,
                dp.box_number,
                r.distance,
                r.venue,
                r.grade,
                r.track_condition,
                r.weather
            FROM dog_performances dp
            JOIN races r ON dp.race_id = r.race_id
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"📈 Loaded {len(df)} records from reference data")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading reference data: {e}")
            raise

    def _calculate_feature_statistics(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for monitored features"""
        stats = {
            "metadata": {
                "model_version": self.model_version,
                "git_sha": self.git_sha,
                "date": self.date,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(df),
                "features_count": len(features)
            },
            "features": {}
        }
        
        for feature in features:
            if feature in df.columns:
                series = df[feature].dropna()
                
                if len(series) > 0:
                    # Basic statistics
                    feature_stats = {
                        "count": len(series),
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "quantiles": {
                            "q25": float(series.quantile(0.25)),
                            "q50": float(series.quantile(0.50)),  # median
                            "q75": float(series.quantile(0.75)),
                            "q90": float(series.quantile(0.90)),
                            "q95": float(series.quantile(0.95))
                        },
                        "outlier_bounds": {
                            "lower": float(series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))),
                            "upper": float(series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))
                        }
                    }
                    
                    # Additional statistics for better monitoring
                    feature_stats["skewness"] = float(series.skew())
                    feature_stats["kurtosis"] = float(series.kurtosis())
                    feature_stats["null_percentage"] = float((len(df[feature]) - len(series)) / len(df[feature]) * 100)
                    
                    stats["features"][feature] = feature_stats
                    
                    logger.debug(f"  📊 {feature}: mean={feature_stats['mean']:.3f}, std={feature_stats['std']:.3f}")
                else:
                    logger.warning(f"⚠️  No valid data for feature: {feature}")
            else:
                logger.warning(f"⚠️  Feature not found in data: {feature}")
        
        return stats

    def create_baseline_stats(self, force_recreate: bool = False) -> Dict[str, Any]:
        """Create baseline statistics from reference data"""
        versioned_dir = self.baseline_dir / self.model_version
        stats_file = versioned_dir / "baseline_stats.json"
        
        # Check if stats already exist
        if stats_file.exists() and not force_recreate:
            logger.info(f"📋 Baseline stats already exist: {stats_file}")
            with open(stats_file, 'r') as f:
                return json.load(f)
        
        # Create versioned directory
        versioned_dir.mkdir(exist_ok=True)
        
        # Load reference data
        logger.info("📊 Loading reference data...")
        df = self._load_reference_data()
        
        # Get monitored features
        features = self.get_monitored_features()
        
        # Calculate statistics
        logger.info(f"🔢 Calculating statistics for {len(features)} features...")
        stats = self._calculate_feature_statistics(df, features)
        
        # Save statistics
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Create a convenience symlink to latest stats
        latest_link = self.baseline_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(versioned_dir.name)
        
        logger.info(f"✅ Baseline statistics created: {stats_file}")
        logger.info(f"🔗 Latest symlink updated: {latest_link}")
        
        return stats

    def load_baseline_stats(self, model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load baseline statistics for a specific model version
        If model_version is None, loads current version or creates if missing
        """
        if model_version is None:
            model_version = self.model_version
        
        stats_file = self.baseline_dir / model_version / "baseline_stats.json"
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                logger.info(f"📊 Loaded baseline stats for version: {model_version}")
                return stats
            except Exception as e:
                logger.error(f"❌ Error loading baseline stats: {e}")
                return None
        else:
            logger.warning(f"📋 Baseline stats not found for version: {model_version}")
            if model_version == self.model_version:
                logger.info("🔄 Creating baseline stats from reference data...")
                return self.create_baseline_stats()
            return None

    def list_available_versions(self) -> List[str]:
        """List all available baseline statistics versions"""
        versions = []
        if self.baseline_dir.exists():
            for item in self.baseline_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name != 'latest':
                    versions.append(item.name)
        return sorted(versions, reverse=True)  # Most recent first

    def get_feature_baseline(self, feature_name: str, model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get baseline statistics for a specific feature"""
        stats = self.load_baseline_stats(model_version)
        if stats and "features" in stats:
            return stats["features"].get(feature_name)
        return None

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare baseline statistics between two versions"""
        stats1 = self.load_baseline_stats(version1)
        stats2 = self.load_baseline_stats(version2)
        
        if not stats1 or not stats2:
            return {"error": "Could not load statistics for comparison"}
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "feature_changes": {}
        }
        
        # Compare features that exist in both versions
        common_features = set(stats1.get("features", {}).keys()) & set(stats2.get("features", {}).keys())
        
        for feature in common_features:
            f1 = stats1["features"][feature]
            f2 = stats2["features"][feature]
            
            comparison["feature_changes"][feature] = {
                "mean_change": f2["mean"] - f1["mean"],
                "std_change": f2["std"] - f1["std"],
                "sample_count_change": f2["count"] - f1["count"]
            }
        
        return comparison

    def export_stats_summary(self, model_version: Optional[str] = None) -> str:
        """Export a human-readable summary of baseline statistics"""
        stats = self.load_baseline_stats(model_version)
        if not stats:
            return "No statistics available"
        
        summary = f"Baseline Statistics Summary\n"
        summary += f"=" * 50 + "\n"
        summary += f"Model Version: {stats['metadata']['model_version']}\n"
        summary += f"Git SHA: {stats['metadata']['git_sha']}\n"
        summary += f"Date: {stats['metadata']['date']}\n"
        summary += f"Total Samples: {stats['metadata']['total_samples']:,}\n"
        summary += f"Features Count: {stats['metadata']['features_count']}\n\n"
        
        for feature_name, feature_stats in stats.get("features", {}).items():
            summary += f"{feature_name}:\n"
            summary += f"  Mean: {feature_stats['mean']:.3f}\n"
            summary += f"  Std: {feature_stats['std']:.3f}\n"
            summary += f"  Range: [{feature_stats['min']:.3f}, {feature_stats['max']:.3f}]\n"
            summary += f"  Quantiles: Q25={feature_stats['quantiles']['q25']:.3f}, "
            summary += f"Q50={feature_stats['quantiles']['q50']:.3f}, Q75={feature_stats['quantiles']['q75']:.3f}\n"
            summary += f"  Samples: {feature_stats['count']:,}\n\n"
        
        return summary


def main():
    """Example usage of the baseline statistics manager"""
    # Initialize the manager
    manager = BaselineStatsManager()
    
    # Create or load baseline statistics
    print("🔄 Loading/Creating baseline statistics...")
    stats = manager.create_baseline_stats(force_recreate=True)
    
    if stats:
        print(f"✅ Statistics loaded for version: {stats['metadata']['model_version']}")
        print(f"📊 Total samples: {stats['metadata']['total_samples']:,}")
        print(f"🔢 Features: {stats['metadata']['features_count']}")
        
        # Print summary
        print("\n" + manager.export_stats_summary())
        
        # List available versions
        versions = manager.list_available_versions()
        print(f"📋 Available versions: {versions}")
        
    else:
        print("❌ Could not load or create baseline statistics")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
