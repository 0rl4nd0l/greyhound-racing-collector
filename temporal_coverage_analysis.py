#!/usr/bin/env python3
"""
Temporal Coverage & Consistency Analysis for Greyhound Racing Data

This script performs comprehensive temporal analysis including:
1. Timeline of race counts per week/month/year
2. Gap and spike detection for scraping outages or schema changes
3. Monotonicity verification of incremental IDs and timestamps
4. Daylight saving time and timezone anomaly detection
"""

import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns

warnings.filterwarnings("ignore")


class TemporalCoverageAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.analysis_results = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def load_race_data(self):
        """Load race metadata with temporal information"""
        query = """
        SELECT 
            race_id as id,
            race_id,
            venue,
            race_date,
            1 as race_number,
            created_at as extraction_timestamp,
            'legacy' as data_source,
            'completed' as race_status,
            8 as field_size,
            COUNT(*) OVER (PARTITION BY race_date, venue) as races_per_day_venue
        FROM races 
        WHERE race_date IS NOT NULL
        ORDER BY race_date, created_at
        """

        df = pd.read_sql_query(query, self.conn)

        # Convert date columns
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        df["extraction_timestamp"] = pd.to_datetime(
            df["extraction_timestamp"], errors="coerce"
        )

        # Extract temporal components
        df["year"] = df["race_date"].dt.year
        df["month"] = df["race_date"].dt.month
        df["week"] = df["race_date"].dt.isocalendar().week
        df["dayofweek"] = df["race_date"].dt.dayofweek
        df["quarter"] = df["race_date"].dt.quarter

        # Create period identifiers
        df["year_month"] = df["race_date"].dt.to_period("M")
        df["year_week"] = df["race_date"].dt.to_period("W")
        df["year_quarter"] = df["race_date"].dt.to_period("Q")

        return df

    def build_race_timelines(self, df):
        """Build comprehensive race count timelines"""
        print("Building race count timelines...")

        timelines = {}

        # Daily timeline
        timelines["daily"] = (
            df.groupby("race_date")
            .agg(
                {"race_id": "count", "venue": "nunique", "field_size": ["mean", "sum"]}
            )
            .round(2)
        )
        timelines["daily"].columns = [
            "race_count",
            "unique_venues",
            "avg_field_size",
            "total_dogs",
        ]

        # Weekly timeline
        timelines["weekly"] = df.groupby("year_week").agg(
            {"race_id": "count", "venue": "nunique", "race_date": ["min", "max"]}
        )
        timelines["weekly"].columns = [
            "race_count",
            "unique_venues",
            "week_start",
            "week_end",
        ]

        # Monthly timeline
        timelines["monthly"] = df.groupby("year_month").agg(
            {"race_id": "count", "venue": "nunique", "race_date": ["min", "max"]}
        )
        timelines["monthly"].columns = [
            "race_count",
            "unique_venues",
            "month_start",
            "month_end",
        ]

        # Yearly timeline
        timelines["yearly"] = df.groupby("year").agg(
            {"race_id": "count", "venue": "nunique", "race_date": ["min", "max"]}
        )
        timelines["yearly"].columns = [
            "race_count",
            "unique_venues",
            "year_start",
            "year_end",
        ]

        # Venue-specific timeline
        timelines["by_venue"] = (
            df.groupby(["venue", "year_month"])
            .agg({"race_id": "count"})
            .unstack(fill_value=0)
        )

        self.analysis_results["timelines"] = timelines
        return timelines

    def detect_gaps_and_spikes(self, df):
        """Identify gaps and unusual spikes in race data"""
        print("Detecting gaps and spikes...")

        daily_counts = df.groupby("race_date")["race_id"].count()

        # Gap detection - days with no races
        date_range = pd.date_range(
            start=daily_counts.index.min(), end=daily_counts.index.max(), freq="D"
        )
        all_dates = pd.Series(0, index=date_range)
        complete_timeline = all_dates.add(daily_counts, fill_value=0)

        gaps = complete_timeline[complete_timeline == 0]

        # Filter out expected gaps (likely weekends or holidays)
        gaps_weekdays = gaps[gaps.index.dayofweek < 5]  # Monday=0, Friday=4

        # Consecutive gap detection
        gap_groups = []
        if len(gaps) > 0:
            gap_dates = gaps.index.tolist()
            current_group = [gap_dates[0]]

            for i in range(1, len(gap_dates)):
                if (gap_dates[i] - gap_dates[i - 1]).days == 1:
                    current_group.append(gap_dates[i])
                else:
                    if len(current_group) > 2:  # Only significant gaps
                        gap_groups.append(
                            {
                                "start": current_group[0],
                                "end": current_group[-1],
                                "duration": len(current_group),
                            }
                        )
                    current_group = [gap_dates[i]]

            if len(current_group) > 2:
                gap_groups.append(
                    {
                        "start": current_group[0],
                        "end": current_group[-1],
                        "duration": len(current_group),
                    }
                )

        # Spike detection using statistical methods
        race_counts = complete_timeline[complete_timeline > 0]
        if len(race_counts) > 0:
            q75, q25 = np.percentile(race_counts, [75, 25])
            iqr = q75 - q25
            upper_threshold = q75 + 2.5 * iqr
            lower_threshold = max(0, q25 - 1.5 * iqr)

            spikes = race_counts[race_counts > upper_threshold]
            unusual_lows = race_counts[race_counts < lower_threshold]
        else:
            spikes = pd.Series(dtype=int)
            unusual_lows = pd.Series(dtype=int)

        # Schema change detection via extraction timestamp analysis
        extraction_gaps = self.detect_extraction_anomalies(df)

        anomalies = {
            "total_gaps": len(gaps),
            "weekday_gaps": len(gaps_weekdays),
            "consecutive_gap_periods": gap_groups,
            "spikes": spikes.to_dict(),
            "unusual_low_days": unusual_lows.to_dict(),
            "extraction_anomalies": extraction_gaps,
            "statistics": {
                "mean_daily_races": race_counts.mean(),
                "std_daily_races": race_counts.std(),
                "median_daily_races": race_counts.median(),
                "spike_threshold": upper_threshold if len(race_counts) > 0 else 0,
            },
        }

        self.analysis_results["anomalies"] = anomalies
        return anomalies

    def detect_extraction_anomalies(self, df):
        """Detect anomalies in extraction timestamps that might indicate scraping issues"""
        if df["extraction_timestamp"].isna().all():
            return {"error": "No extraction timestamps available"}

        # Group by extraction date and analyze patterns
        df_extract = df.dropna(subset=["extraction_timestamp"])
        df_extract["extraction_date"] = df_extract["extraction_timestamp"].dt.date

        daily_extractions = df_extract.groupby("extraction_date").agg(
            {"race_id": "count", "extraction_timestamp": ["min", "max"]}
        )
        daily_extractions.columns = [
            "records_extracted",
            "first_extraction",
            "last_extraction",
        ]

        # Detect unusual extraction patterns
        extraction_counts = daily_extractions["records_extracted"]
        if len(extraction_counts) > 0:
            q75, q25 = np.percentile(extraction_counts, [75, 25])
            iqr = q75 - q25
            upper_threshold = q75 + 2 * iqr

            unusual_extraction_days = extraction_counts[
                extraction_counts > upper_threshold
            ]
        else:
            unusual_extraction_days = pd.Series(dtype=int)

        return {
            "unusual_extraction_days": unusual_extraction_days.to_dict(),
            "extraction_date_range": {
                "start": df_extract["extraction_timestamp"].min(),
                "end": df_extract["extraction_timestamp"].max(),
            },
        }

    def verify_monotonicity(self, df):
        """Verify monotonicity of IDs and timestamps to catch retroactive edits"""
        print("Verifying monotonicity of IDs and timestamps...")

        monotonicity_issues = {}

        # Check ID monotonicity
        id_sequence = df.sort_values("id")["id"].tolist()
        id_gaps = []
        id_duplicates = []

        for i in range(1, len(id_sequence)):
            if id_sequence[i] <= id_sequence[i - 1]:
                id_duplicates.append(
                    {
                        "position": i,
                        "current_id": id_sequence[i],
                        "previous_id": id_sequence[i - 1],
                    }
                )
            elif id_sequence[i] - id_sequence[i - 1] > 1:
                gap_size = id_sequence[i] - id_sequence[i - 1] - 1
                if gap_size > 10:  # Only report significant gaps
                    id_gaps.append(
                        {
                            "after_id": id_sequence[i - 1],
                            "before_id": id_sequence[i],
                            "gap_size": gap_size,
                        }
                    )

        # Check timestamp monotonicity relative to race dates
        df_time_sorted = df.dropna(
            subset=["race_date", "extraction_timestamp"]
        ).sort_values("extraction_timestamp")

        timestamp_issues = []
        for i in range(1, len(df_time_sorted)):
            current_race_date = df_time_sorted.iloc[i]["race_date"]
            previous_race_date = df_time_sorted.iloc[i - 1]["race_date"]
            current_extract_time = df_time_sorted.iloc[i]["extraction_timestamp"]
            previous_extract_time = df_time_sorted.iloc[i - 1]["extraction_timestamp"]

            # Check for retroactive data (race date goes backwards while extraction time goes forward)
            if (
                current_race_date < previous_race_date
                and current_extract_time > previous_extract_time
            ):
                timestamp_issues.append(
                    {
                        "index": i,
                        "race_id": df_time_sorted.iloc[i]["race_id"],
                        "race_date": current_race_date,
                        "previous_race_date": previous_race_date,
                        "extraction_time": current_extract_time,
                        "issue_type": "retroactive_data",
                    }
                )

        monotonicity_issues = {
            "id_gaps": id_gaps,
            "id_duplicates": id_duplicates,
            "timestamp_anomalies": timestamp_issues,
            "total_records": len(df),
            "id_range": {"min": df["id"].min(), "max": df["id"].max()},
            "extraction_time_range": {
                "min": df["extraction_timestamp"].min(),
                "max": df["extraction_timestamp"].max(),
            },
        }

        self.analysis_results["monotonicity"] = monotonicity_issues
        return monotonicity_issues

    def check_timezone_anomalies(self, df):
        """Check for daylight saving time and timezone anomalies"""
        print("Checking timezone and daylight saving anomalies...")

        if df["extraction_timestamp"].isna().all():
            return {"error": "No extraction timestamps for timezone analysis"}

        df_tz = df.dropna(subset=["extraction_timestamp"]).copy()

        # Assume data is in Australian timezone (common for greyhound racing)
        # Check for DST transitions
        au_tz = pytz.timezone("Australia/Sydney")

        timezone_issues = []
        dst_transitions = []

        # Group by day and check for unusual time patterns
        df_tz["extract_hour"] = df_tz["extraction_timestamp"].dt.hour
        daily_extract_patterns = df_tz.groupby(
            df_tz["extraction_timestamp"].dt.date
        ).agg({"extract_hour": ["min", "max", "std"], "race_id": "count"})
        daily_extract_patterns.columns = [
            "min_hour",
            "max_hour",
            "hour_std",
            "record_count",
        ]

        # Detect unusual hour patterns (possible timezone shifts)
        unusual_hours = daily_extract_patterns[
            (daily_extract_patterns["hour_std"] > 6)
            | (  # High variation in extraction hours
                daily_extract_patterns["min_hour"] < 3
            )
            | (  # Very early extractions
                daily_extract_patterns["max_hour"] > 23
            )  # Very late extractions
        ]

        # Check for DST transition dates (rough approximation)
        # In Australia, DST typically changes in October and April
        potential_dst_dates = df_tz[
            (
                (df_tz["extraction_timestamp"].dt.month == 10)
                & (df_tz["extraction_timestamp"].dt.day.between(1, 7))
            )
            | (
                (df_tz["extraction_timestamp"].dt.month == 4)
                & (df_tz["extraction_timestamp"].dt.day.between(1, 7))
            )
        ]["extraction_timestamp"].dt.date.unique()

        timezone_analysis = {
            "unusual_extraction_patterns": unusual_hours.to_dict("index"),
            "potential_dst_transition_dates": [str(d) for d in potential_dst_dates],
            "extraction_hour_distribution": df_tz["extract_hour"]
            .value_counts()
            .to_dict(),
            "time_span_analysis": {
                "total_days": (
                    df_tz["extraction_timestamp"].max()
                    - df_tz["extraction_timestamp"].min()
                ).days,
                "unique_extraction_dates": df_tz[
                    "extraction_timestamp"
                ].dt.date.nunique(),
            },
        }

        self.analysis_results["timezone"] = timezone_analysis
        return timezone_analysis

    def generate_visualizations(self, df):
        """Generate temporal analysis visualizations"""
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Temporal Coverage Analysis", fontsize=16)

        # Daily race counts over time
        daily_counts = df.groupby("race_date")["race_id"].count()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, alpha=0.7)
        axes[0, 0].set_title("Daily Race Counts Over Time")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Number of Races")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Monthly aggregation
        monthly_counts = df.groupby("year_month")["race_id"].count()
        axes[0, 1].bar(range(len(monthly_counts)), monthly_counts.values, alpha=0.7)
        axes[0, 1].set_title("Monthly Race Counts")
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Number of Races")

        # Day of week distribution
        dow_counts = df.groupby("dayofweek")["race_id"].count()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        axes[1, 0].bar(day_names, dow_counts.values, alpha=0.7)
        axes[1, 0].set_title("Races by Day of Week")
        axes[1, 0].set_ylabel("Number of Races")

        # Venue distribution over time
        venue_timeline = (
            df.groupby(["year_month", "venue"])["race_id"].count().unstack(fill_value=0)
        )
        if venue_timeline.shape[1] <= 10:  # Only plot if manageable number of venues
            venue_timeline.plot(kind="area", stacked=True, ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_title("Race Distribution by Venue Over Time")
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Show top venues only
            top_venues = df["venue"].value_counts().head(10).index
            venue_timeline_top = venue_timeline[top_venues]
            venue_timeline_top.plot(kind="area", stacked=True, ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_title(f"Race Distribution by Top {len(top_venues)} Venues")
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig("./temporal_coverage_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("Visualizations saved as 'temporal_coverage_analysis.png'")

    def run_full_analysis(self):
        """Run complete temporal coverage analysis"""
        print("Starting comprehensive temporal coverage analysis...")
        print("=" * 60)

        # Load data
        df = self.load_race_data()
        print(f"Loaded {len(df)} race records")
        print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        print(f"Unique venues: {df['venue'].nunique()}")
        print()

        # Build timelines
        timelines = self.build_race_timelines(df)

        # Detect anomalies
        anomalies = self.detect_gaps_and_spikes(df)

        # Verify monotonicity
        monotonicity = self.verify_monotonicity(df)

        # Check timezone issues
        timezone_analysis = self.check_timezone_anomalies(df)

        # Generate visualizations
        self.generate_visualizations(df)

        return self.analysis_results

    def print_summary_report(self):
        """Print a comprehensive summary report"""
        print("\n" + "=" * 80)
        print("TEMPORAL COVERAGE & CONSISTENCY ANALYSIS REPORT")
        print("=" * 80)

        results = self.analysis_results

        # Timeline Summary
        if "timelines" in results:
            print("\n📊 TIMELINE SUMMARY")
            print("-" * 40)
            timelines = results["timelines"]

            if "yearly" in timelines:
                print("Yearly Race Counts:")
                for year, data in timelines["yearly"].iterrows():
                    print(
                        f"  {year}: {data['race_count']} races across {data['unique_venues']} venues"
                    )

            if "monthly" in timelines:
                recent_months = timelines["monthly"].tail(6)
                print(f"\nRecent Monthly Activity (last 6 months):")
                for month, data in recent_months.iterrows():
                    print(
                        f"  {month}: {data['race_count']} races, {data['unique_venues']} venues"
                    )

        # Anomaly Summary
        if "anomalies" in results:
            print("\n🚨 ANOMALY DETECTION")
            print("-" * 40)
            anomalies = results["anomalies"]

            print(f"Total gaps in data: {anomalies['total_gaps']}")
            print(f"Weekday gaps (unusual): {anomalies['weekday_gaps']}")
            print(
                f"Consecutive gap periods: {len(anomalies['consecutive_gap_periods'])}"
            )

            if anomalies["consecutive_gap_periods"]:
                print("  Significant gap periods:")
                for gap in anomalies["consecutive_gap_periods"][:5]:
                    print(
                        f"    {gap['start']} to {gap['end']} ({gap['duration']} days)"
                    )

            print(f"Unusual spike days: {len(anomalies['spikes'])}")
            if anomalies["spikes"]:
                print("  Top spike days:")
                spikes_sorted = sorted(
                    anomalies["spikes"].items(), key=lambda x: x[1], reverse=True
                )
                for date, count in spikes_sorted[:5]:
                    print(f"    {date}: {count} races")

            stats = anomalies["statistics"]
            print(f"\nRace Count Statistics:")
            print(f"  Average daily races: {stats['mean_daily_races']:.1f}")
            print(f"  Standard deviation: {stats['std_daily_races']:.1f}")
            print(f"  Median daily races: {stats['median_daily_races']:.1f}")

        # Monotonicity Summary
        if "monotonicity" in results:
            print("\n🔢 MONOTONICITY VERIFICATION")
            print("-" * 40)
            mono = results["monotonicity"]

            print(f"ID range: {mono['id_range']['min']} to {mono['id_range']['max']}")
            print(f"ID gaps detected: {len(mono['id_gaps'])}")
            print(f"ID duplicate issues: {len(mono['id_duplicates'])}")
            print(f"Timestamp anomalies: {len(mono['timestamp_anomalies'])}")

            if mono["id_gaps"]:
                print("  Significant ID gaps:")
                for gap in mono["id_gaps"][:3]:
                    print(
                        f"    Gap of {gap['gap_size']} between IDs {gap['after_id']} and {gap['before_id']}"
                    )

            if mono["timestamp_anomalies"]:
                print("  Timestamp issues:")
                for issue in mono["timestamp_anomalies"][:3]:
                    print(f"    Race {issue['race_id']}: {issue['issue_type']}")

        # Timezone Summary
        if "timezone" in results:
            print("\n🌐 TIMEZONE & DST ANALYSIS")
            print("-" * 40)
            tz = results["timezone"]

            if "error" not in tz:
                print(
                    f"Potential DST transition dates: {len(tz['potential_dst_transition_dates'])}"
                )
                print(
                    f"Unusual extraction patterns: {len(tz['unusual_extraction_patterns'])}"
                )

                time_span = tz["time_span_analysis"]
                print(f"Data collection span: {time_span['total_days']} days")
                print(
                    f"Unique extraction dates: {time_span['unique_extraction_dates']}"
                )

                # Show extraction hour distribution
                hour_dist = tz["extraction_hour_distribution"]
                if hour_dist:
                    most_common_hour = max(hour_dist, key=hour_dist.get)
                    print(
                        f"Most common extraction hour: {most_common_hour}:00 ({hour_dist[most_common_hour]} records)"
                    )
            else:
                print(f"Timezone analysis error: {tz['error']}")

        print("\n" + "=" * 80)
        print(
            "Analysis complete! Check 'temporal_coverage_analysis.png' for visualizations."
        )
        print("=" * 80)


def main():
    """Main analysis execution"""
    database_path = "./databases/race_data.db"

    try:
        with TemporalCoverageAnalyzer(database_path) as analyzer:
            results = analyzer.run_full_analysis()
            analyzer.print_summary_report()

            # Save simplified results to avoid JSON serialization issues
            import json

            simplified_results = {
                "summary": {
                    "total_records": len(results.get("timelines", {}).get("daily", [])),
                    "date_range": {
                        "start": str(
                            results.get("monotonicity", {})
                            .get("extraction_time_range", {})
                            .get("min", "")
                        ),
                        "end": str(
                            results.get("monotonicity", {})
                            .get("extraction_time_range", {})
                            .get("max", "")
                        ),
                    },
                    "anomalies_found": {
                        "gaps": results.get("anomalies", {}).get("total_gaps", 0),
                        "spikes": len(results.get("anomalies", {}).get("spikes", {})),
                        "id_gaps": len(
                            results.get("monotonicity", {}).get("id_gaps", [])
                        ),
                        "timestamp_issues": len(
                            results.get("monotonicity", {}).get(
                                "timestamp_anomalies", []
                            )
                        ),
                    },
                }
            }

            with open("./temporal_analysis_summary.json", "w") as f:
                json.dump(simplified_results, f, indent=2)

            print(f"\nSummary results saved to 'temporal_analysis_summary.json'")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
