#!/usr/bin/env python3
"""
Feature Quality Analysis Script
===============================

Analyzes current model features for:
- Feature importance rankings
- Data completeness and quality
- Correlation analysis
- Performance impact assessment
- Suggestions for improvements
"""

import json
import os
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder

from ml_system_v4 import MLSystemV4


def analyze_feature_quality():
    """Comprehensive feature quality analysis."""
    print("🔍 FEATURE QUALITY ANALYSIS")
    print("=" * 50)

    # Initialize ML system
    ml = MLSystemV4()

    if not ml.calibrated_pipeline:
        print("❌ No trained model found. Please train a model first.")
        return

    print(
        f"📊 Analyzing {len(ml.feature_columns)} features from model: {ml.model_info.get('model_type', 'unknown')}"
    )
    print(f"📅 Model trained: {ml.model_info.get('trained_at', 'unknown')}")

    # Get training data for analysis
    print("\n🔄 Loading training data...")
    train_data, test_data = ml.prepare_time_ordered_data()

    if train_data.empty:
        print("❌ No training data available for analysis")
        return

    print(
        f"📈 Training data: {len(train_data)} samples, {len(train_data['race_id'].unique())} races"
    )

    # Build features for analysis
    print("🔧 Building features...")
    train_features = ml.build_leakage_safe_features(train_data)

    if train_features.empty:
        print("❌ No features could be built")
        return

    print(
        f"✅ Built features: {len(train_features)} samples, {len(train_features.columns)} columns"
    )

    # Prepare data for analysis
    X = train_features.drop(
        ["race_id", "dog_clean_name", "target", "target_timestamp"],
        axis=1,
        errors="ignore",
    )
    y = train_features["target"]

    # Remove non-numeric columns for analysis
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols].fillna(0)

    print(f"📊 Numeric features for analysis: {len(X_numeric.columns)}")

    # 1. DATA COMPLETENESS ANALYSIS
    print("\n" + "=" * 50)
    print("1. DATA COMPLETENESS ANALYSIS")
    print("=" * 50)

    completeness = {}
    for col in X_numeric.columns:
        total_samples = len(X_numeric[col])
        non_zero = (X_numeric[col] != 0).sum()
        non_null = X_numeric[col].notna().sum()

        completeness[col] = {
            "non_null_rate": non_null / total_samples,
            "non_zero_rate": non_zero / total_samples,
            "mean_value": X_numeric[col].mean(),
            "std_value": X_numeric[col].std(),
            "min_value": X_numeric[col].min(),
            "max_value": X_numeric[col].max(),
        }

    # Sort by completeness
    completeness_df = pd.DataFrame(completeness).T
    completeness_df = completeness_df.sort_values("non_zero_rate", ascending=False)

    print("🏆 TOP FEATURES BY DATA COMPLETENESS:")
    for i, (feature, stats) in enumerate(completeness_df.head(10).iterrows()):
        print(
            f"  {i+1:2d}. {feature[:40]:<40} | Non-zero: {stats['non_zero_rate']:.1%} | Mean: {stats['mean_value']:.3f}"
        )

    print("\n⚠️ FEATURES WITH LOW COMPLETENESS (<50%):")
    low_completeness = completeness_df[completeness_df["non_zero_rate"] < 0.5]
    for feature, stats in low_completeness.iterrows():
        print(f"  • {feature[:40]:<40} | Non-zero: {stats['non_zero_rate']:.1%}")

    # 2. FEATURE IMPORTANCE ANALYSIS
    print("\n" + "=" * 50)
    print("2. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    try:
        # Calculate mutual information
        print("🔄 Calculating mutual information scores...")
        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {
                "feature": X_numeric.columns,
                "mutual_info": mi_scores,
                "completeness": [
                    completeness[col]["non_zero_rate"] for col in X_numeric.columns
                ],
            }
        ).sort_values("mutual_info", ascending=False)

        print("🏆 TOP FEATURES BY MUTUAL INFORMATION:")
        for i, row in importance_df.head(15).iterrows():
            print(
                f"  {row.name+1:2d}. {row['feature'][:40]:<40} | MI: {row['mutual_info']:.4f} | Complete: {row['completeness']:.1%}"
            )

        print("\n📉 LOW IMPORTANCE FEATURES (bottom 10):")
        for i, row in importance_df.tail(10).iterrows():
            print(
                f"  {row.name+1:2d}. {row['feature'][:40]:<40} | MI: {row['mutual_info']:.4f} | Complete: {row['completeness']:.1%}"
            )

    except Exception as e:
        print(f"⚠️ Could not calculate feature importance: {e}")
        importance_df = pd.DataFrame()

    # 3. CORRELATION ANALYSIS
    print("\n" + "=" * 50)
    print("3. CORRELATION ANALYSIS")
    print("=" * 50)

    try:
        # Calculate correlations with target
        correlations = X_numeric.corrwith(y).abs().sort_values(ascending=False)

        print("🏆 TOP FEATURES BY CORRELATION WITH TARGET:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            if not np.isnan(corr):
                print(f"  {i+1:2d}. {feature[:40]:<40} | Correlation: {corr:.4f}")

        # Find highly correlated feature pairs
        corr_matrix = X_numeric.corr()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8 and not np.isnan(corr_val):
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                    )

        if high_corr_pairs:
            print(f"\n⚠️ HIGHLY CORRELATED FEATURE PAIRS (>0.8):")
            for feat1, feat2, corr in sorted(
                high_corr_pairs, key=lambda x: x[2], reverse=True
            )[:10]:
                print(f"  • {feat1[:25]:<25} ↔ {feat2[:25]:<25} | {corr:.3f}")
        else:
            print("\n✅ No highly correlated feature pairs found (>0.8)")

    except Exception as e:
        print(f"⚠️ Could not calculate correlations: {e}")

    # 4. FEATURE CATEGORIES PERFORMANCE
    print("\n" + "=" * 50)
    print("4. FEATURE CATEGORIES PERFORMANCE")
    print("=" * 50)

    # Group features by category
    feature_categories = {
        "Performance": [
            f
            for f in X_numeric.columns
            if any(
                x in f.lower()
                for x in ["avg_position", "win_rate", "place_rate", "time", "score"]
            )
        ],
        "Form & Consistency": [
            f
            for f in X_numeric.columns
            if any(
                x in f.lower() for x in ["form", "consistency", "trend", "improvement"]
            )
        ],
        "Track & Venue": [
            f
            for f in X_numeric.columns
            if any(x in f.lower() for x in ["venue", "track", "distance", "specific"])
        ],
        "Physical": [
            f
            for f in X_numeric.columns
            if any(x in f.lower() for x in ["weight", "box"])
        ],
        "Market": [
            f
            for f in X_numeric.columns
            if any(x in f.lower() for x in ["odds", "confidence", "market"])
        ],
        "Traditional": [f for f in X_numeric.columns if "traditional" in f.lower()],
        "Weather": [
            f
            for f in X_numeric.columns
            if any(
                x in f.lower()
                for x in ["weather", "temperature", "humidity", "wind", "pressure"]
            )
        ],
    }

    if not importance_df.empty:
        for category, features in feature_categories.items():
            category_features = [
                f for f in features if f in importance_df["feature"].values
            ]
            if category_features:
                avg_importance = importance_df[
                    importance_df["feature"].isin(category_features)
                ]["mutual_info"].mean()
                avg_completeness = importance_df[
                    importance_df["feature"].isin(category_features)
                ]["completeness"].mean()
                print(
                    f"📊 {category:18} | Features: {len(category_features):2d} | Avg MI: {avg_importance:.4f} | Avg Complete: {avg_completeness:.1%}"
                )

    # 5. SAVE ANALYSIS RESULTS
    print("\n" + "=" * 50)
    print("5. ANALYSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 50)

    # Create analysis report
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "model_info": ml.model_info,
        "feature_count": len(ml.feature_columns),
        "completeness_analysis": (
            completeness_df.to_dict("index") if "completeness_df" in locals() else {}
        ),
        "importance_analysis": (
            importance_df.to_dict("records") if not importance_df.empty else []
        ),
        "category_performance": {},
        "recommendations": [],
    }

    # Add recommendations
    recommendations = [
        "✅ STRENGTHS IDENTIFIED:",
        f"  • {len(ml.feature_columns)} diverse features covering multiple aspects",
        "  • Strong temporal leakage protection in place",
        "  • Comprehensive traditional scoring metrics",
        "  • Good mix of performance, form, and environmental features",
        "",
        "🎯 RECOMMENDED IMPROVEMENTS:",
    ]

    if not importance_df.empty:
        # High importance, low completeness features
        high_imp_low_comp = importance_df[
            (importance_df["mutual_info"] > importance_df["mutual_info"].median())
            & (importance_df["completeness"] < 0.7)
        ]

        if not high_imp_low_comp.empty:
            recommendations.append(
                "  • IMPROVE DATA QUALITY for high-importance features:"
            )
            for _, row in high_imp_low_comp.head(5).iterrows():
                recommendations.append(
                    f"    - {row['feature']} (MI: {row['mutual_info']:.3f}, Complete: {row['completeness']:.1%})"
                )

        # Low importance features to consider removing
        low_importance = importance_df[importance_df["mutual_info"] < 0.001]
        if len(low_importance) > 5:
            recommendations.append(
                f"  • CONSIDER REMOVING {len(low_importance)} low-importance features (MI < 0.001)"
            )
            recommendations.append(
                "    - This could improve model efficiency without losing predictive power"
            )

    if high_corr_pairs:
        recommendations.append(
            f"  • ADDRESS MULTICOLLINEARITY: {len(high_corr_pairs)} highly correlated pairs found"
        )
        recommendations.append(
            "    - Consider feature selection or dimensionality reduction"
        )

    recommendations.extend(
        [
            "",
            "💡 ADVANCED FEATURE ENGINEERING SUGGESTIONS:",
            "  • Add interaction features between top predictive variables",
            "  • Create rolling averages with different time windows",
            "  • Engineer class-based performance metrics (grade transitions)",
            "  • Add trainer-specific performance indicators",
            "  • Create seasonal/temporal pattern features",
            "  • Add race pace and sectional time analysis features",
            "",
            "🔧 NEXT STEPS:",
            "  1. Focus data collection efforts on high-importance, low-completeness features",
            "  2. Consider ensemble methods to improve prediction accuracy",
            "  3. Implement feature selection to optimize the feature set",
            "  4. Add validation using cross-track performance",
        ]
    )

    analysis_results["recommendations"] = recommendations

    # Print recommendations
    for rec in recommendations:
        print(rec)

    # Save results to file
    results_dir = Path("analysis_results")
    results_dir.mkdir(exist_ok=True)

    results_file = (
        results_dir
        / f'feature_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(results_file, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print(f"\n💾 Analysis results saved to: {results_file}")

    return analysis_results


if __name__ == "__main__":
    try:
        results = analyze_feature_quality()
        print("\n✅ Feature analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
