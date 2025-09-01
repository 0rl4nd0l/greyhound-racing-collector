#!/usr/bin/env python3
"""
Baseline Statistics Management API
==================================

This module implements the exact requirements for Step 9: Baseline statistics management.

Key Features:
- Load `baseline_stats/{model_version}.json`; if missing create from reference_data and save
- Include mean, std, quantiles for monitored features
- Version directory with model sha/date to guarantee traceability

Usage:
    from baseline_stats_api import load_or_create_baseline_stats
    
    # Load or create baseline statistics for current model version
    stats = load_or_create_baseline_stats()
    
    # Load baseline statistics for specific model version
    stats = load_or_create_baseline_stats(model_version="157602b_20250803")

Author: AI Assistant
Date: August 3, 2025
"""

import json
import logging
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_current_model_version() -> str:
    """
    Generate current model version using git SHA and date for traceability
    Format: {git_sha}_{date}
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not get git SHA, using timestamp")
        git_sha = datetime.now().strftime("%H%M%S")

    date = datetime.now().strftime("%Y%m%d")
    return f"{git_sha}_{date}"


def load_reference_data(
    database_path: str = "./databases/race_data.db",
) -> pd.DataFrame:
    """
    Load reference data from the database for baseline statistics calculation
    """
    try:
        conn = sqlite3.connect(database_path)

        # Query to get monitored features from reference data
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


def calculate_feature_statistics(
    df: pd.DataFrame, features: List[str], model_version: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics (mean, std, quantiles) for monitored features
    """
    git_sha, date = model_version.split("_", 1)

    stats = {
        "metadata": {
            "model_version": model_version,
            "git_sha": git_sha,
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df),
            "features_count": len(features),
        },
        "features": {},
    }

    for feature in features:
        if feature in df.columns:
            series = df[feature].dropna()

            if len(series) > 0:
                # Calculate required statistics: mean, std, quantiles
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
                        "q95": float(series.quantile(0.95)),
                    },
                }

                # Additional monitoring statistics
                feature_stats["skewness"] = float(series.skew())
                feature_stats["kurtosis"] = float(series.kurtosis())
                feature_stats["null_percentage"] = float(
                    (len(df[feature]) - len(series)) / len(df[feature]) * 100
                )

                stats["features"][feature] = feature_stats

                logger.info(
                    f"  📊 {feature}: mean={feature_stats['mean']:.3f}, std={feature_stats['std']:.3f}"
                )
            else:
                logger.warning(f"⚠️  No valid data for feature: {feature}")
        else:
            logger.warning(f"⚠️  Feature not found in data: {feature}")

    return stats


def get_monitored_features() -> List[str]:
    """
    Define the features to monitor for baseline statistics
    These should align with your model's input features
    """
    return [
        "weight",
        "box_number",
        # Add more features as they become available in the data
        # "distance",
        # "race_time",
        # "finish_position"
    ]


def load_or_create_baseline_stats(
    model_version: Optional[str] = None,
    baseline_dir: str = "./baseline_stats",
    reference_db_path: str = "./databases/race_data.db",
    force_recreate: bool = False,
) -> Dict[str, Any]:
    """
    Load `baseline_stats/{model_version}.json`; if missing create from reference_data and save.
    Include mean, std, quantiles for monitored features.
    Version directory with model sha/date to guarantee traceability.

    Args:
        model_version: Model version string (git_sha_date). If None, uses current version.
        baseline_dir: Directory to store baseline statistics
        reference_db_path: Path to reference database
        force_recreate: Force recreation even if stats file exists

    Returns:
        Dictionary containing baseline statistics with metadata and feature statistics
    """

    # Get model version for traceability
    if model_version is None:
        model_version = get_current_model_version()

    # Create versioned directory path for traceability
    baseline_path = Path(baseline_dir)
    versioned_dir = baseline_path / model_version
    stats_file = versioned_dir / f"{model_version}.json"

    logger.info(f"📊 Processing baseline stats for model version: {model_version}")

    # Load existing stats if they exist and not forcing recreation
    if stats_file.exists() and not force_recreate:
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
            logger.info(f"✅ Loaded existing baseline stats: {stats_file}")
            return stats
        except Exception as e:
            logger.error(f"❌ Error loading existing stats, recreating: {e}")

    # Create baseline statistics from reference data
    logger.info("🔄 Creating baseline statistics from reference data...")

    # Ensure versioned directory exists
    versioned_dir.mkdir(parents=True, exist_ok=True)

    # Load reference data
    df = load_reference_data(reference_db_path)

    # Get monitored features
    features = get_monitored_features()

    # Calculate statistics
    logger.info(f"🔢 Calculating statistics for {len(features)} features...")
    stats = calculate_feature_statistics(df, features, model_version)

    # Save statistics to versioned directory
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Create convenience symlink to latest stats
    latest_link = baseline_path / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(model_version)
        logger.info(f"🔗 Latest symlink updated: {latest_link}")
    except OSError:
        # Fallback for systems that don't support symlinks
        logger.warning("Could not create symlink, skipping...")

    logger.info(f"✅ Baseline statistics created: {stats_file}")
    logger.info(
        f"📊 Features: {len(stats['features'])}, Samples: {stats['metadata']['total_samples']:,}"
    )

    return stats


def get_baseline_stats_summary(stats: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of baseline statistics
    """
    if not stats:
        return "No statistics available"

    metadata = stats.get("metadata", {})
    features = stats.get("features", {})

    summary = []
    summary.append(f"Baseline Statistics Summary")
    summary.append(f"=" * 50)
    summary.append(f"Model Version: {metadata.get('model_version', 'Unknown')}")
    summary.append(f"Git SHA: {metadata.get('git_sha', 'Unknown')}")
    summary.append(f"Date: {metadata.get('date', 'Unknown')}")
    summary.append(f"Total Samples: {metadata.get('total_samples', 0):,}")
    summary.append(f"Features Count: {len(features)}")
    summary.append("")

    for feature_name, feature_stats in features.items():
        summary.append(f"{feature_name}:")
        summary.append(f"  Mean: {feature_stats['mean']:.3f}")
        summary.append(f"  Std: {feature_stats['std']:.3f}")
        summary.append(
            f"  Range: [{feature_stats['min']:.3f}, {feature_stats['max']:.3f}]"
        )
        summary.append(
            f"  Quantiles: Q25={feature_stats['quantiles']['q25']:.3f}, "
            f"Q50={feature_stats['quantiles']['q50']:.3f}, Q75={feature_stats['quantiles']['q75']:.3f}"
        )
        summary.append(f"  Samples: {feature_stats['count']:,}")
        summary.append("")

    return "\n".join(summary)


def main():
    """
    Example usage demonstrating the baseline statistics management
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🚀 Baseline Statistics Management - Step 9 Implementation")
    print("=" * 60)

    # Load or create baseline statistics for current model version
    print("\n1. Loading/Creating baseline statistics for current model version...")
    stats = load_or_create_baseline_stats()

    if stats:
        print("\n2. Statistics Summary:")
        print(get_baseline_stats_summary(stats))

        # Demonstrate loading specific version
        model_version = stats["metadata"]["model_version"]
        print(f"3. Loading specific model version: {model_version}")
        specific_stats = load_or_create_baseline_stats(model_version=model_version)

        if specific_stats:
            print(f"✅ Successfully loaded stats for version: {model_version}")
            print(
                f"📁 File location: baseline_stats/{model_version}/{model_version}.json"
            )

        # Show directory structure
        print(f"\n4. Directory structure:")
        baseline_dir = Path("./baseline_stats")
        if baseline_dir.exists():
            for item in baseline_dir.iterdir():
                if item.is_dir():
                    files = list(item.glob("*.json"))
                    print(f"  📁 {item.name}/ -> {len(files)} files")
                elif item.is_symlink():
                    print(f"  🔗 {item.name} -> {item.readlink()}")
    else:
        print("❌ Failed to load or create baseline statistics")


if __name__ == "__main__":
    main()
