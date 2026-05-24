#!/usr/bin/env python3
"""
Statistical Completeness & Distribution Analysis for Greyhound Racing Data

This script performs comprehensive analysis of:
1. Numeric fields: histograms, boxplots, summary stats, skewness, outliers, modality
2. Categorical fields: frequency tables, entropy, sparsity analysis
3. Data completeness assessment
4. Model training readiness evaluation
"""

import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import entropy, kurtosis, skew

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class StatisticalAnalyzer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.results = {}

    def load_data(self):
        """Load data from both main tables"""
        print("Loading data from database...")

        # Load race metadata
        self.race_data = pd.read_sql_query(
            """
            SELECT * FROM race_metadata 
            WHERE race_status != 'cancelled' OR race_status IS NULL
        """,
            self.conn,
        )

        # Load dog race data
        self.dog_data = pd.read_sql_query(
            """
            SELECT * FROM dog_race_data 
            WHERE was_scratched != 1 OR was_scratched IS NULL
        """,
            self.conn,
        )

        print(
            f"Loaded {len(self.race_data)} race records and {len(self.dog_data)} dog performance records"
        )

        # Merge for comprehensive analysis
        self.merged_data = pd.merge(
            self.dog_data, self.race_data, on="race_id", how="left"
        )

        print(f"Created merged dataset with {len(self.merged_data)} records")

    def analyze_completeness(self, df, name):
        """Analyze data completeness"""
        print(f"\n{'='*50}")
        print(f"DATA COMPLETENESS ANALYSIS - {name}")
        print(f"{'='*50}")

        total_rows = len(df)
        completeness_stats = {}

        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / total_rows) * 100
            completeness_stats[col] = {
                "null_count": null_count,
                "null_percentage": null_pct,
                "completeness": 100 - null_pct,
            }

        # Sort by completeness
        sorted_completeness = sorted(
            completeness_stats.items(), key=lambda x: x[1]["completeness"]
        )

        print(f"\nCOMPLETENESS SUMMARY (Total rows: {total_rows})")
        print("-" * 80)
        print(f"{'Field':<35} {'Missing':<10} {'Missing %':<12} {'Complete %':<12}")
        print("-" * 80)

        for field, stats in sorted_completeness:
            print(
                f"{field:<35} {stats['null_count']:<10} {stats['null_percentage']:<12.2f} {stats['completeness']:<12.2f}"
            )

        # Flag critical missing data
        critical_missing = [
            (field, stats)
            for field, stats in completeness_stats.items()
            if stats["null_percentage"] > 50
        ]

        if critical_missing:
            print(f"\n⚠️  CRITICAL MISSING DATA (>50% missing):")
            for field, stats in critical_missing:
                print(f"   • {field}: {stats['null_percentage']:.1f}% missing")

        return completeness_stats

    def identify_numeric_fields(self, df):
        """Identify numeric fields for analysis"""
        numeric_fields = []

        # Primary numeric fields of interest
        target_numeric_fields = [
            "weight",
            "odds_decimal",
            "starting_price",
            "beaten_margin",
            "performance_rating",
            "speed_rating",
            "class_rating",
            "win_probability",
            "place_probability",
            "best_time",
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",
            "visibility",
            "precipitation",
            "winner_odds",
            "winner_margin",
            "prize_money_total",
            "field_size",
            "actual_field_size",
            "scratch_rate",
            "finish_position",
            "placing",
        ]

        for field in target_numeric_fields:
            if field in df.columns:
                # Check if field is actually numeric
                try:
                    pd.to_numeric(df[field], errors="coerce")
                    numeric_fields.append(field)
                except Exception:
                    continue

        return numeric_fields

    def analyze_numeric_distribution(self, df, field):
        """Comprehensive numeric field analysis"""
        data = pd.to_numeric(df[field], errors="coerce").dropna()

        if len(data) == 0:
            return None

        # Basic statistics
        stats_dict = {
            "count": len(data),
            "mean": data.mean(),
            "median": data.median(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "q25": data.quantile(0.25),
            "q75": data.quantile(0.75),
            "iqr": data.quantile(0.75) - data.quantile(0.25),
            "range": data.max() - data.min(),
        }

        # Distribution characteristics
        stats_dict["skewness"] = skew(data)
        stats_dict["kurtosis"] = kurtosis(data)

        # Outlier detection using IQR method
        iqr = stats_dict["iqr"]
        q1 = stats_dict["q25"]
        q3 = stats_dict["q75"]
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        stats_dict["outlier_count"] = len(outliers)
        stats_dict["outlier_percentage"] = (len(outliers) / len(data)) * 100

        # Normality test (Shapiro-Wilk for samples < 5000)
        if len(data) < 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                stats_dict["shapiro_statistic"] = shapiro_stat
                stats_dict["shapiro_p_value"] = shapiro_p
                stats_dict["is_normal"] = shapiro_p > 0.05
            except Exception:
                stats_dict["is_normal"] = None
        else:
            stats_dict["is_normal"] = None

        # Modality assessment using Hartigan's dip test approximation
        # For simplicity, we'll use coefficient of variation and examine histogram peaks
        cv = (
            stats_dict["std"] / abs(stats_dict["mean"])
            if stats_dict["mean"] != 0
            else float("inf")
        )
        stats_dict["coefficient_of_variation"] = cv

        return stats_dict

    def create_numeric_plots(self, df, field, stats_dict, output_dir="analysis_plots"):
        """Create visualization plots for numeric fields"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        data = pd.to_numeric(df[field], errors="coerce").dropna()

        if len(data) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Distribution Analysis: {field}", fontsize=16, fontweight="bold")

        # Histogram
        axes[0, 0].hist(data, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].axvline(
            data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.2f}"
        )
        axes[0, 0].axvline(
            data.median(),
            color="green",
            linestyle="--",
            label=f"Median: {data.median():.2f}",
        )
        axes[0, 0].set_title("Histogram")
        axes[0, 0].set_xlabel(field)
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Box plot
        axes[0, 1].boxplot([data], labels=[field])
        axes[0, 1].set_title("Box Plot")
        axes[0, 1].set_ylabel("Values")
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot for normality assessment
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
        axes[1, 0].grid(True, alpha=0.3)

        # Statistics text
        stats_text = f"""
        Count: {stats_dict['count']:,}
        Mean: {stats_dict['mean']:.4f}
        Median: {stats_dict['median']:.4f}
        Std Dev: {stats_dict['std']:.4f}
        
        Skewness: {stats_dict['skewness']:.4f}
        Kurtosis: {stats_dict['kurtosis']:.4f}
        
        Outliers: {stats_dict['outlier_count']} ({stats_dict['outlier_percentage']:.2f}%)
        CV: {stats_dict['coefficient_of_variation']:.4f}
        """

        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        )
        axes[1, 1].set_title("Summary Statistics")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{field}_distribution_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def identify_categorical_fields(self, df):
        """Identify categorical fields for analysis"""
        categorical_fields = []

        # Primary categorical fields of interest
        target_categorical_fields = [
            "venue",
            "grade",
            "track_condition",
            "weather",
            "wind_direction",
            "running_style",
            "trainer_name",
            "dog_name",
            "race_name",
            "data_source",
            "weather_condition",
            "venue_slug",
            "form",
            "scraped_race_classification",
        ]

        for field in target_categorical_fields:
            if field in df.columns:
                categorical_fields.append(field)

        # Also include fields with object dtype that aren't clearly numeric
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in categorical_fields and col not in [
                "race_id",
                "extraction_timestamp",
            ]:
                # Check if it's not a numeric field stored as string
                try:
                    numeric_count = (
                        pd.to_numeric(df[col], errors="coerce").notna().sum()
                    )
                    total_count = df[col].notna().sum()
                    if (
                        total_count > 0 and numeric_count / total_count < 0.8
                    ):  # Less than 80% numeric
                        categorical_fields.append(col)
                except Exception:
                    categorical_fields.append(col)

        return categorical_fields

    def analyze_categorical_distribution(self, df, field):
        """Comprehensive categorical field analysis"""
        data = df[field].dropna()

        if len(data) == 0:
            return None

        # Frequency analysis
        value_counts = data.value_counts()
        total_count = len(data)

        # Basic statistics
        stats_dict = {
            "total_count": total_count,
            "unique_values": len(value_counts),
            "most_common_value": (
                value_counts.index[0] if len(value_counts) > 0 else None
            ),
            "most_common_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
            "most_common_percentage": (
                (value_counts.iloc[0] / total_count * 100)
                if len(value_counts) > 0
                else 0
            ),
        }

        # Entropy calculation (information content)
        probabilities = value_counts / total_count
        stats_dict["entropy"] = entropy(probabilities, base=2)
        stats_dict["max_entropy"] = (
            np.log2(len(value_counts)) if len(value_counts) > 0 else 0
        )
        stats_dict["normalized_entropy"] = (
            (stats_dict["entropy"] / stats_dict["max_entropy"])
            if stats_dict["max_entropy"] > 0
            else 0
        )

        # Sparsity analysis
        singleton_count = (value_counts == 1).sum()  # Values appearing only once
        sparse_threshold = (
            0.01 * total_count
        )  # Values appearing in less than 1% of records
        sparse_count = (value_counts < sparse_threshold).sum()

        stats_dict["singleton_count"] = singleton_count
        stats_dict["singleton_percentage"] = (
            (singleton_count / len(value_counts) * 100) if len(value_counts) > 0 else 0
        )
        stats_dict["sparse_count"] = sparse_count
        stats_dict["sparse_percentage"] = (
            (sparse_count / len(value_counts) * 100) if len(value_counts) > 0 else 0
        )

        # Balance analysis
        gini_coefficient = self.calculate_gini_coefficient(value_counts.values)
        stats_dict["gini_coefficient"] = gini_coefficient

        # Top categories
        top_10 = value_counts.head(10)
        stats_dict["top_categories"] = list(
            zip(top_10.index, top_10.values, (top_10.values / total_count * 100))
        )

        return stats_dict

    def calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for measuring inequality in distribution"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    def create_categorical_plots(
        self, df, field, stats_dict, output_dir="analysis_plots"
    ):
        """Create visualization plots for categorical fields"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        data = df[field].dropna()

        if len(data) == 0:
            return

        value_counts = data.value_counts()

        # Limit to top 20 categories for readability
        top_categories = value_counts.head(20)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Categorical Analysis: {field}", fontsize=16, fontweight="bold")

        # Bar plot of top categories
        top_categories.plot(kind="bar", ax=axes[0, 0], color="lightcoral")
        axes[0, 0].set_title("Top 20 Categories by Frequency")
        axes[0, 0].set_xlabel("Categories")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Pie chart of top 10 categories
        top_10 = value_counts.head(10)
        other_count = value_counts.iloc[10:].sum() if len(value_counts) > 10 else 0

        if other_count > 0:
            pie_data = list(top_10.values) + [other_count]
            pie_labels = list(top_10.index) + ["Others"]
        else:
            pie_data = top_10.values
            pie_labels = top_10.index

        axes[0, 1].pie(pie_data, labels=pie_labels, autopct="%1.1f%%", startangle=90)
        axes[0, 1].set_title("Distribution (Top 10 + Others)")

        # Distribution histogram (category rank vs frequency)
        ranks = np.arange(1, len(value_counts) + 1)
        axes[1, 0].plot(ranks, value_counts.values, "o-", alpha=0.7)
        axes[1, 0].set_title("Rank-Frequency Distribution")
        axes[1, 0].set_xlabel("Category Rank")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Statistics text
        stats_text = f"""
        Total Values: {stats_dict['total_count']:,}
        Unique Categories: {stats_dict['unique_values']:,}
        
        Most Common: {stats_dict['most_common_value']}
        ({stats_dict['most_common_percentage']:.1f}%)
        
        Entropy: {stats_dict['entropy']:.4f}
        Normalized Entropy: {stats_dict['normalized_entropy']:.4f}
        
        Sparse Categories: {stats_dict['sparse_count']} ({stats_dict['sparse_percentage']:.1f}%)
        Singletons: {stats_dict['singleton_count']} ({stats_dict['singleton_percentage']:.1f}%)
        
        Gini Coefficient: {stats_dict['gini_coefficient']:.4f}
        """

        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        )
        axes[1, 1].set_title("Summary Statistics")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{field}_categorical_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def flag_model_training_issues(self, numeric_results, categorical_results):
        """Flag potential issues for model training"""
        print(f"\n{'='*60}")
        print("MODEL TRAINING READINESS ASSESSMENT")
        print(f"{'='*60}")

        issues = []
        warnings = []

        # Numeric field issues
        for field, stats in numeric_results.items():
            if stats is None:
                continue

            # High skewness
            if abs(stats["skewness"]) > 2:
                issues.append(
                    f"🔴 {field}: Highly skewed (skewness: {stats['skewness']:.2f})"
                )
            elif abs(stats["skewness"]) > 1:
                warnings.append(
                    f"🟡 {field}: Moderately skewed (skewness: {stats['skewness']:.2f})"
                )

            # High outlier percentage
            if stats["outlier_percentage"] > 10:
                issues.append(
                    f"🔴 {field}: High outlier rate ({stats['outlier_percentage']:.1f}%)"
                )
            elif stats["outlier_percentage"] > 5:
                warnings.append(
                    f"🟡 {field}: Moderate outlier rate ({stats['outlier_percentage']:.1f}%)"
                )

            # High coefficient of variation
            if stats["coefficient_of_variation"] > 2:
                warnings.append(
                    f"🟡 {field}: High variability (CV: {stats['coefficient_of_variation']:.2f})"
                )

        # Categorical field issues
        for field, stats in categorical_results.items():
            if stats is None:
                continue

            # High sparsity
            if stats["sparse_percentage"] > 50:
                issues.append(
                    f"🔴 {field}: Very sparse ({stats['sparse_percentage']:.1f}% categories are rare)"
                )
            elif stats["sparse_percentage"] > 25:
                warnings.append(
                    f"🟡 {field}: Moderately sparse ({stats['sparse_percentage']:.1f}% categories are rare)"
                )

            # High cardinality
            if stats["unique_values"] > 1000:
                issues.append(
                    f"🔴 {field}: Very high cardinality ({stats['unique_values']:,} unique values)"
                )
            elif stats["unique_values"] > 100:
                warnings.append(
                    f"🟡 {field}: High cardinality ({stats['unique_values']:,} unique values)"
                )

            # Highly imbalanced
            if stats["most_common_percentage"] > 90:
                issues.append(
                    f"🔴 {field}: Highly imbalanced ({stats['most_common_percentage']:.1f}% in one category)"
                )
            elif stats["most_common_percentage"] > 80:
                warnings.append(
                    f"🟡 {field}: Moderately imbalanced ({stats['most_common_percentage']:.1f}% in one category)"
                )

            # Low entropy (low information content)
            if stats["normalized_entropy"] < 0.1:
                warnings.append(
                    f"🟡 {field}: Low information content (entropy: {stats['normalized_entropy']:.3f})"
                )

        # Display results
        if issues:
            print("\n🚨 CRITICAL ISSUES (require attention before training):")
            for issue in issues:
                print(f"   {issue}")

        if warnings:
            print("\n⚠️  WARNINGS (consider addressing):")
            for warning in warnings:
                print(f"   {warning}")

        if not issues and not warnings:
            print(
                "\n✅ No major issues detected - data appears ready for model training!"
            )

        return issues, warnings

    def generate_summary_report(
        self,
        completeness_results,
        numeric_results,
        categorical_results,
        issues,
        warnings,
    ):
        """Generate comprehensive summary report"""
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL COMPLETENESS & DISTRIBUTION ANALYSIS SUMMARY")
        report.append("=" * 80)

        # Data overview
        report.append(f"\nDATA OVERVIEW:")
        report.append(f"• Race records: {len(self.race_data):,}")
        report.append(f"• Dog performance records: {len(self.dog_data):,}")
        report.append(f"• Merged dataset size: {len(self.merged_data):,}")

        # Completeness summary
        report.append(f"\nCOMPLETENESS SUMMARY:")
        complete_fields = len(
            [
                f
                for f, s in completeness_results["merged"].items()
                if s["completeness"] >= 90
            ]
        )
        total_fields = len(completeness_results["merged"])
        report.append(
            f"• Fields with ≥90% completeness: {complete_fields}/{total_fields} ({(complete_fields/total_fields)*100:.1f}%)"
        )

        critical_missing = len(
            [
                f
                for f, s in completeness_results["merged"].items()
                if s["null_percentage"] > 50
            ]
        )
        if critical_missing > 0:
            report.append(f"• Fields with >50% missing data: {critical_missing}")

        # Numeric analysis summary
        report.append(f"\nNUMERIC FIELDS ANALYSIS:")
        valid_numeric = len([f for f, s in numeric_results.items() if s is not None])
        report.append(f"• Analyzed numeric fields: {valid_numeric}")

        skewed_fields = len(
            [f for f, s in numeric_results.items() if s and abs(s["skewness"]) > 1]
        )
        report.append(f"• Fields with significant skewness: {skewed_fields}")

        outlier_fields = len(
            [f for f, s in numeric_results.items() if s and s["outlier_percentage"] > 5]
        )
        report.append(f"• Fields with notable outliers (>5%): {outlier_fields}")

        # Categorical analysis summary
        report.append(f"\nCATEGORICAL FIELDS ANALYSIS:")
        valid_categorical = len(
            [f for f, s in categorical_results.items() if s is not None]
        )
        report.append(f"• Analyzed categorical fields: {valid_categorical}")

        high_cardinality = len(
            [
                f
                for f, s in categorical_results.items()
                if s and s["unique_values"] > 100
            ]
        )
        report.append(
            f"• High cardinality fields (>100 categories): {high_cardinality}"
        )

        sparse_fields = len(
            [
                f
                for f, s in categorical_results.items()
                if s and s["sparse_percentage"] > 25
            ]
        )
        report.append(f"• Sparse fields (>25% rare categories): {sparse_fields}")

        imbalanced_fields = len(
            [
                f
                for f, s in categorical_results.items()
                if s and s["most_common_percentage"] > 80
            ]
        )
        report.append(
            f"• Highly imbalanced fields (>80% in one category): {imbalanced_fields}"
        )

        # Issues summary
        report.append(f"\nMODEL TRAINING READINESS:")
        report.append(f"• Critical issues: {len(issues)}")
        report.append(f"• Warnings: {len(warnings)}")

        if len(issues) == 0:
            report.append("• Overall assessment: READY for model training")
        elif len(issues) < 5:
            report.append("• Overall assessment: NEEDS MINOR FIXES before training")
        else:
            report.append(
                "• Overall assessment: NEEDS SIGNIFICANT PREPROCESSING before training"
            )

        return "\n".join(report)

    def run_complete_analysis(self):
        """Run complete statistical analysis"""
        print("Starting comprehensive statistical analysis...")

        # Load data
        self.load_data()

        # Analyze completeness for all datasets
        completeness_results = {}
        completeness_results["race_data"] = self.analyze_completeness(
            self.race_data, "RACE METADATA"
        )
        completeness_results["dog_data"] = self.analyze_completeness(
            self.dog_data, "DOG PERFORMANCE DATA"
        )
        completeness_results["merged"] = self.analyze_completeness(
            self.merged_data, "MERGED DATASET"
        )

        # Identify and analyze numeric fields
        print(f"\n{'='*50}")
        print("NUMERIC FIELDS ANALYSIS")
        print(f"{'='*50}")

        numeric_fields = self.identify_numeric_fields(self.merged_data)
        print(
            f"Identified {len(numeric_fields)} numeric fields: {', '.join(numeric_fields)}"
        )

        numeric_results = {}
        for field in numeric_fields:
            print(f"\nAnalyzing numeric field: {field}")
            stats = self.analyze_numeric_distribution(self.merged_data, field)
            numeric_results[field] = stats

            if stats:
                print(f"  • Count: {stats['count']:,}")
                print(f"  • Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")
                print(f"  • Skewness: {stats['skewness']:.4f}")
                print(
                    f"  • Outliers: {stats['outlier_count']} ({stats['outlier_percentage']:.2f}%)"
                )

                # Create plots
                self.create_numeric_plots(self.merged_data, field, stats)

        # Identify and analyze categorical fields
        print(f"\n{'='*50}")
        print("CATEGORICAL FIELDS ANALYSIS")
        print(f"{'='*50}")

        categorical_fields = self.identify_categorical_fields(self.merged_data)
        print(
            f"Identified {len(categorical_fields)} categorical fields: {', '.join(categorical_fields)}"
        )

        categorical_results = {}
        for field in categorical_fields:
            print(f"\nAnalyzing categorical field: {field}")
            stats = self.analyze_categorical_distribution(self.merged_data, field)
            categorical_results[field] = stats

            if stats:
                print(f"  • Total values: {stats['total_count']:,}")
                print(f"  • Unique categories: {stats['unique_values']:,}")
                print(
                    f"  • Entropy: {stats['entropy']:.4f} (normalized: {stats['normalized_entropy']:.4f})"
                )
                print(
                    f"  • Most common: {stats['most_common_value']} ({stats['most_common_percentage']:.1f}%)"
                )
                print(
                    f"  • Sparse categories: {stats['sparse_count']} ({stats['sparse_percentage']:.1f}%)"
                )

                # Create plots
                self.create_categorical_plots(self.merged_data, field, stats)

        # Flag model training issues
        issues, warnings = self.flag_model_training_issues(
            numeric_results, categorical_results
        )

        # Generate summary report
        summary_report = self.generate_summary_report(
            completeness_results, numeric_results, categorical_results, issues, warnings
        )

        print(f"\n{summary_report}")

        # Save results
        self.save_results(
            {
                "completeness": completeness_results,
                "numeric_analysis": numeric_results,
                "categorical_analysis": categorical_results,
                "issues": issues,
                "warnings": warnings,
                "summary_report": summary_report,
            }
        )

        print(
            f"\n✅ Analysis complete! Results saved to 'statistical_analysis_results.json'"
        )
        print("📊 Visualization plots saved to 'analysis_plots/' directory")

    def save_results(self, results):
        """Save analysis results to JSON file"""
        import json

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results_clean = convert_numpy_types(results)

        with open("statistical_analysis_results.json", "w") as f:
            json.dump(results_clean, f, indent=2, default=str)

    def __del__(self):
        """Close database connection"""
        if hasattr(self, "conn"):
            self.conn.close()


if __name__ == "__main__":
    analyzer = StatisticalAnalyzer()
    analyzer.run_complete_analysis()
