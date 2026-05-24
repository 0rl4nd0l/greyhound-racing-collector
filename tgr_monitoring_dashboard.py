#!/usr/bin/env python3
"""
TGR System Monitoring & Alerting Dashboard
==========================================

Real-time monitoring of TGR data collection and system health
with alerting for issues and performance optimization recommendations.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TGRMonitoringDashboard:
    """Comprehensive monitoring dashboard for TGR system health."""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.alerts = []
        self.metrics = {}

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive TGR system report."""

        logger.info("🔍 Generating comprehensive TGR system report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self._assess_system_health(),
            "data_quality": self._assess_data_quality(),
            "performance_metrics": self._calculate_performance_metrics(),
            "cache_efficiency": self._analyze_cache_efficiency(),
            "collection_trends": self._analyze_collection_trends(),
            "alerts": self._generate_alerts(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall TGR system health."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            health = {"status": "healthy", "components": {}, "uptime_metrics": {}}

            # Check table health
            tables = [
                "tgr_dog_performance_summary",
                "tgr_expert_insights",
                "tgr_enhanced_dog_form",
                "tgr_enhanced_feature_cache",
                "tgr_scraping_log",
            ]

            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]

                    # Check for recent activity
                    cursor.execute(
                        f"""
                        SELECT COUNT(*) FROM {table} 
                        WHERE (created_at >= datetime('now', '-24 hours') 
                               OR last_updated >= datetime('now', '-24 hours')
                               OR cached_at >= datetime('now', '-24 hours'))
                    """
                    )
                    recent_activity = cursor.fetchone()[0]

                    health["components"][table] = {
                        "status": "healthy" if count > 0 else "warning",
                        "record_count": count,
                        "recent_activity": recent_activity,
                    }

                except Exception as e:
                    health["components"][table] = {"status": "error", "error": str(e)}

            # Check scraping performance
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_attempts,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successes,
                    AVG(scrape_duration) as avg_duration,
                    MAX(created_at) as last_attempt
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-7 days')
            """
            )

            scraping_stats = cursor.fetchone()
            if scraping_stats and scraping_stats[0] > 0:
                total, successes, avg_duration, last_attempt = scraping_stats
                success_rate = (successes / total) * 100 if total > 0 else 0

                health["uptime_metrics"] = {
                    "success_rate": success_rate,
                    "avg_scrape_duration": avg_duration or 0,
                    "last_activity": last_attempt,
                    "total_attempts_7d": total,
                }

                # Overall health assessment
                if success_rate < 50:
                    health["status"] = "critical"
                elif success_rate < 75:
                    health["status"] = "warning"

            conn.close()
            return health

        except Exception as e:
            logger.error(f"Health assessment error: {e}")
            return {"status": "error", "error": str(e)}

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess TGR data quality metrics."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            quality = {
                "overall_score": 0,
                "completeness": {},
                "consistency": {},
                "freshness": {},
            }

            # Completeness metrics
            cursor.execute("SELECT COUNT(*) FROM tgr_dog_performance_summary")
            total_dogs = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM tgr_dog_performance_summary 
                WHERE total_entries > 0 AND wins >= 0 AND places >= 0
            """
            )
            complete_profiles = cursor.fetchone()[0]

            completeness_score = (
                (complete_profiles / total_dogs * 100) if total_dogs > 0 else 0
            )

            quality["completeness"] = {
                "total_dogs": total_dogs,
                "complete_profiles": complete_profiles,
                "score": completeness_score,
            }

            # Data freshness
            cursor.execute(
                """
                SELECT 
                    COUNT(CASE WHEN last_updated >= datetime('now', '-7 days') THEN 1 END) as recent,
                    COUNT(CASE WHEN last_updated >= datetime('now', '-30 days') THEN 1 END) as monthly,
                    COUNT(*) as total
                FROM tgr_dog_performance_summary
            """
            )

            freshness_stats = cursor.fetchone()
            recent, monthly, total = freshness_stats

            freshness_score = (recent / total * 100) if total > 0 else 0

            quality["freshness"] = {
                "recent_updates_7d": recent,
                "recent_updates_30d": monthly,
                "total_records": total,
                "freshness_score": freshness_score,
            }

            # Consistency checks
            cursor.execute(
                """
                SELECT 
                    COUNT(CASE WHEN win_percentage BETWEEN 0 AND 100 THEN 1 END) as valid_win_rates,
                    COUNT(CASE WHEN consistency_score BETWEEN 0 AND 100 THEN 1 END) as valid_consistency,
                    COUNT(*) as total
                FROM tgr_dog_performance_summary
            """
            )

            consistency_stats = cursor.fetchone()
            valid_wins, valid_consistency, total = consistency_stats

            consistency_score = (
                ((valid_wins + valid_consistency) / (2 * total) * 100)
                if total > 0
                else 0
            )

            quality["consistency"] = {
                "valid_win_rates": valid_wins,
                "valid_consistency": valid_consistency,
                "score": consistency_score,
            }

            # Overall quality score
            quality["overall_score"] = (
                completeness_score + freshness_score + consistency_score
            ) / 3

            conn.close()
            return quality

        except Exception as e:
            logger.error(f"Data quality assessment error: {e}")
            return {"overall_score": 0, "error": str(e)}

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate TGR system performance metrics."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metrics = {}

            # Feature cache performance
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_cache_entries,
                    COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as valid_cache,
                    AVG(julianday('now') - julianday(cached_at)) as avg_cache_age_days
                FROM tgr_enhanced_feature_cache
            """
            )

            cache_stats = cursor.fetchone()
            total_cache, valid_cache, avg_age = cache_stats

            cache_hit_rate = (valid_cache / total_cache * 100) if total_cache > 0 else 0

            metrics["cache_performance"] = {
                "total_entries": total_cache,
                "valid_entries": valid_cache,
                "hit_rate": cache_hit_rate,
                "avg_age_days": avg_age or 0,
            }

            # Collection efficiency
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_collections,
                    AVG(entries_found) as avg_entries_per_collection,
                    AVG(comments_found) as avg_comments_per_collection,
                    AVG(scrape_duration) as avg_collection_time
                FROM tgr_scraping_log
                WHERE created_at >= datetime('now', '-30 days')
            """
            )

            collection_stats = cursor.fetchone()
            if collection_stats and collection_stats[0] > 0:
                total_collections, avg_entries, avg_comments, avg_time = (
                    collection_stats
                )

                metrics["collection_efficiency"] = {
                    "total_collections_30d": total_collections,
                    "avg_entries_per_collection": avg_entries or 0,
                    "avg_comments_per_collection": avg_comments or 0,
                    "avg_collection_time_sec": avg_time or 0,
                    "collections_per_day": total_collections / 30,
                }

            # Performance trends
            cursor.execute(
                """
                SELECT 
                    COUNT(CASE WHEN form_trend = 'improving' THEN 1 END) as improving,
                    COUNT(CASE WHEN form_trend = 'stable' THEN 1 END) as stable,
                    COUNT(CASE WHEN form_trend = 'declining' THEN 1 END) as declining,
                    AVG(win_percentage) as avg_win_rate,
                    AVG(consistency_score) as avg_consistency
                FROM tgr_dog_performance_summary
            """
            )

            trend_stats = cursor.fetchone()
            improving, stable, declining, avg_win, avg_consistency = trend_stats

            metrics["performance_trends"] = {
                "dogs_improving": improving or 0,
                "dogs_stable": stable or 0,
                "dogs_declining": declining or 0,
                "avg_win_rate": avg_win or 0,
                "avg_consistency": avg_consistency or 0,
            }

            conn.close()
            return metrics

        except Exception as e:
            logger.error(f"Performance metrics calculation error: {e}")
            return {"error": str(e)}

    def _analyze_cache_efficiency(self) -> Dict[str, Any]:
        """Analyze TGR feature cache efficiency."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Cache utilization analysis
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as active_entries,
                    COUNT(CASE WHEN expires_at <= datetime('now') THEN 1 END) as expired_entries,
                    AVG(length(tgr_features)) as avg_feature_size,
                    MIN(cached_at) as oldest_entry,
                    MAX(cached_at) as newest_entry
                FROM tgr_enhanced_feature_cache
            """
            )

            cache_analysis = cursor.fetchone()
            total, active, expired, avg_size, oldest, newest = cache_analysis

            efficiency = {
                "utilization": {
                    "total_entries": total or 0,
                    "active_entries": active or 0,
                    "expired_entries": expired or 0,
                    "utilization_rate": (active / total * 100) if total > 0 else 0,
                },
                "storage": {
                    "avg_feature_size_bytes": avg_size or 0,
                    "estimated_total_size_mb": (
                        (total * (avg_size or 0)) / (1024 * 1024)
                        if total and avg_size
                        else 0
                    ),
                },
                "temporal": {
                    "oldest_entry": oldest,
                    "newest_entry": newest,
                    "cache_span_days": 0,
                },
            }

            # Calculate cache span
            if oldest and newest:
                from datetime import datetime

                oldest_dt = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
                newest_dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
                span = (newest_dt - oldest_dt).days
                efficiency["temporal"]["cache_span_days"] = span

            # Cache miss analysis
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM tgr_scraping_log 
                WHERE status = 'success' 
                AND created_at >= datetime('now', '-7 days')
            """
            )
            recent_collections = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(DISTINCT dog_name) 
                FROM tgr_enhanced_feature_cache 
                WHERE cached_at >= datetime('now', '-7 days')
            """
            )
            recent_cache_entries = cursor.fetchone()[0]

            cache_generation_rate = (
                (recent_cache_entries / recent_collections)
                if recent_collections > 0
                else 0
            )

            efficiency["performance"] = {
                "recent_collections_7d": recent_collections,
                "recent_cache_entries_7d": recent_cache_entries,
                "cache_generation_rate": cache_generation_rate,
            }

            conn.close()
            return efficiency

        except Exception as e:
            logger.error(f"Cache efficiency analysis error: {e}")
            return {"error": str(e)}

    def _analyze_collection_trends(self) -> Dict[str, Any]:
        """Analyze TGR data collection trends over time."""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            trends = {}

            # Daily collection volume trends
            cursor.execute(
                """
                SELECT 
                    date(created_at) as collection_date,
                    COUNT(*) as daily_collections,
                    AVG(entries_found) as avg_entries,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successes
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-30 days')
                GROUP BY date(created_at)
                ORDER BY collection_date DESC
                LIMIT 7
            """
            )

            daily_trends = cursor.fetchall()

            trends["daily_activity"] = []
            for date, collections, avg_entries, successes in daily_trends:
                success_rate = (successes / collections * 100) if collections > 0 else 0
                trends["daily_activity"].append(
                    {
                        "date": date,
                        "collections": collections,
                        "avg_entries": avg_entries or 0,
                        "success_rate": success_rate,
                    }
                )

            # Weekly performance comparison
            cursor.execute(
                """
                SELECT 
                    'this_week' as period,
                    COUNT(*) as collections,
                    AVG(entries_found) as avg_entries,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-7 days')
                UNION ALL
                SELECT 
                    'last_week' as period,
                    COUNT(*) as collections,
                    AVG(entries_found) as avg_entries,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-14 days') 
                AND created_at < datetime('now', '-7 days')
            """
            )

            weekly_comparison = cursor.fetchall()
            trends["weekly_comparison"] = {}

            for period, collections, avg_entries, success_rate in weekly_comparison:
                trends["weekly_comparison"][period] = {
                    "collections": collections or 0,
                    "avg_entries": avg_entries or 0,
                    "success_rate": success_rate or 0,
                }

            # Calculate week-over-week changes
            if (
                "this_week" in trends["weekly_comparison"]
                and "last_week" in trends["weekly_comparison"]
            ):
                this_week = trends["weekly_comparison"]["this_week"]
                last_week = trends["weekly_comparison"]["last_week"]

                trends["week_over_week"] = {
                    "collections_change": this_week["collections"]
                    - last_week["collections"],
                    "entries_change": this_week["avg_entries"]
                    - last_week["avg_entries"],
                    "success_rate_change": this_week["success_rate"]
                    - last_week["success_rate"],
                }

            conn.close()
            return trends

        except Exception as e:
            logger.error(f"Collection trends analysis error: {e}")
            return {"error": str(e)}

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts based on monitoring data."""

        alerts = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Alert: Low success rate
            cursor.execute(
                """
                SELECT COUNT(CASE WHEN status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-24 hours')
            """
            )

            result = cursor.fetchone()
            if result and result[0] is not None:
                success_rate = result[0]
                if success_rate < 50:
                    alerts.append(
                        {
                            "level": "critical",
                            "type": "low_success_rate",
                            "message": f"TGR collection success rate is critically low: {success_rate:.1f}%",
                            "recommendation": "Check network connectivity and TGR website availability",
                        }
                    )
                elif success_rate < 75:
                    alerts.append(
                        {
                            "level": "warning",
                            "type": "moderate_success_rate",
                            "message": f"TGR collection success rate is below normal: {success_rate:.1f}%",
                            "recommendation": "Monitor collection patterns and consider retry logic adjustment",
                        }
                    )

            # Alert: Stale cache entries
            cursor.execute(
                """
                SELECT COUNT(*) as expired_entries
                FROM tgr_enhanced_feature_cache 
                WHERE expires_at <= datetime('now')
            """
            )

            expired_count = cursor.fetchone()[0]
            if expired_count > 50:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "stale_cache",
                        "message": f"{expired_count} feature cache entries have expired",
                        "recommendation": "Run cache maintenance or refresh stale entries",
                    }
                )

            # Alert: No recent activity
            cursor.execute(
                """
                SELECT MAX(created_at) as last_activity
                FROM tgr_scraping_log
            """
            )

            last_activity = cursor.fetchone()[0]
            if last_activity:
                from datetime import datetime

                last_dt = datetime.fromisoformat(last_activity)
                hours_since = (datetime.now() - last_dt).total_seconds() / 3600

                if hours_since > 48:
                    alerts.append(
                        {
                            "level": "warning",
                            "type": "no_recent_activity",
                            "message": f"No TGR collection activity for {hours_since:.1f} hours",
                            "recommendation": "Check if collection processes are running",
                        }
                    )

            # Alert: Data quality issues
            cursor.execute(
                """
                SELECT COUNT(*) as invalid_data
                FROM tgr_dog_performance_summary 
                WHERE win_percentage < 0 OR win_percentage > 100 
                OR consistency_score < 0 OR consistency_score > 100
            """
            )

            invalid_count = cursor.fetchone()[0]
            if invalid_count > 0:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "data_quality",
                        "message": f"{invalid_count} records have invalid performance data",
                        "recommendation": "Review data validation rules and fix corrupted entries",
                    }
                )

            conn.close()

        except Exception as e:
            logger.error(f"Alert generation error: {e}")
            alerts.append(
                {
                    "level": "critical",
                    "type": "system_error",
                    "message": f"Monitoring system error: {str(e)}",
                    "recommendation": "Check database connectivity and monitoring system health",
                }
            )

        return alerts

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system optimization recommendations."""

        recommendations = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Recommendation: Cache optimization
            cursor.execute(
                """
                SELECT COUNT(*) as total, 
                       COUNT(CASE WHEN expires_at <= datetime('now') THEN 1 END) as expired
                FROM tgr_enhanced_feature_cache
            """
            )

            total, expired = cursor.fetchone()
            if total > 0:
                expiry_rate = expired / total * 100
                if expiry_rate > 30:
                    recommendations.append(
                        {
                            "type": "cache_optimization",
                            "priority": "medium",
                            "message": f"{expiry_rate:.1f}% of cache entries are expired",
                            "action": "Consider increasing cache duration or implementing background refresh",
                        }
                    )

            # Recommendation: Collection frequency
            cursor.execute(
                """
                SELECT COUNT(*) as collections_last_7d
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-7 days')
            """
            )

            recent_collections = cursor.fetchone()[0]
            if recent_collections < 10:
                recommendations.append(
                    {
                        "type": "collection_frequency",
                        "priority": "high",
                        "message": f"Only {recent_collections} collections in the last 7 days",
                        "action": "Increase collection frequency for better data freshness",
                    }
                )

            # Recommendation: Data coverage
            cursor.execute(
                """
                SELECT COUNT(DISTINCT dog_name) as dogs_with_tgr_data
                FROM tgr_dog_performance_summary
            """
            )

            cursor.execute(
                """
                SELECT COUNT(DISTINCT dog_clean_name) as total_dogs_in_system
                FROM dog_race_data 
                WHERE dog_clean_name IS NOT NULL AND dog_clean_name != ''
                LIMIT 100
            """
            )

            tgr_dogs = cursor.fetchone()[0]
            total_dogs = cursor.fetchone()[0]

            if total_dogs > 0:
                coverage_rate = tgr_dogs / total_dogs * 100
                if coverage_rate < 50:
                    recommendations.append(
                        {
                            "type": "data_coverage",
                            "priority": "medium",
                            "message": f"TGR data covers only {coverage_rate:.1f}% of dogs in system",
                            "action": "Expand TGR collection to cover more dogs for better ML features",
                        }
                    )

            # Recommendation: Performance optimization
            cursor.execute(
                """
                SELECT AVG(scrape_duration) as avg_duration
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-7 days') AND status = 'success'
            """
            )

            avg_duration = cursor.fetchone()[0]
            if avg_duration and avg_duration > 10:
                recommendations.append(
                    {
                        "type": "performance_optimization",
                        "priority": "low",
                        "message": f"Average collection time is {avg_duration:.1f} seconds",
                        "action": "Consider optimizing scraping logic or implementing parallel collection",
                    }
                )

            conn.close()

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            recommendations.append(
                {
                    "type": "system_maintenance",
                    "priority": "high",
                    "message": "Monitoring system encountered errors",
                    "action": "Review system logs and ensure database connectivity",
                }
            )

        return recommendations

    def print_dashboard(self):
        """Print a formatted monitoring dashboard to console."""

        report = self.generate_comprehensive_report()

        print("🚨 TGR System Monitoring Dashboard")
        print("=" * 60)
        print(f"📅 Report Time: {report['timestamp']}")

        # System Health
        health = report["system_health"]
        health_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "🚨",
            "error": "❌",
        }.get(health.get("status"), "❓")

        print(
            f"\n{health_emoji} System Status: {health.get('status', 'unknown').upper()}"
        )

        if "uptime_metrics" in health and health["uptime_metrics"]:
            metrics = health["uptime_metrics"]
            print(f"   Success Rate: {metrics.get('success_rate', 0):.1f}%")
            print(f"   Avg Duration: {metrics.get('avg_scrape_duration', 0):.1f}s")
            print(f"   Last Activity: {metrics.get('last_activity', 'Unknown')}")

        # Data Quality
        quality = report["data_quality"]
        quality_score = quality.get("overall_score", 0)
        quality_emoji = (
            "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
        )

        print(f"\n{quality_emoji} Data Quality Score: {quality_score:.1f}/100")
        if "completeness" in quality:
            comp = quality["completeness"]
            print(
                f"   Completeness: {comp.get('complete_profiles', 0)}/{comp.get('total_dogs', 0)} dogs"
            )

        # Alerts
        alerts = report["alerts"]
        if alerts:
            print(f"\n🚨 ALERTS ({len(alerts)}):")
            for alert in alerts[:3]:  # Show top 3 alerts
                level_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
                    alert.get("level"), "❓"
                )
                print(f"   {level_emoji} {alert.get('message', 'Unknown alert')}")
        else:
            print(f"\n✅ No active alerts")

        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(recommendations)}):")
            for rec in recommendations[:2]:  # Show top 2 recommendations
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    rec.get("priority"), "⚪"
                )
                print(
                    f"   {priority_emoji} {rec.get('message', 'Unknown recommendation')}"
                )

        print("\n" + "=" * 60)

    def export_report(self, filepath: str = None):
        """Export comprehensive report to JSON file."""

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"tgr_monitoring_report_{timestamp}.json"

        report = self.generate_comprehensive_report()

        try:
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ Monitoring report exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return None


def main():
    """Main monitoring dashboard execution."""

    dashboard = TGRMonitoringDashboard()

    print("🔍 TGR System Health Check")
    print("-" * 30)

    # Display dashboard
    dashboard.print_dashboard()

    # Export detailed report
    report_file = dashboard.export_report()
    if report_file:
        print(f"\n📄 Detailed report saved to: {report_file}")

    print("\n🔧 Monitoring complete!")


if __name__ == "__main__":
    main()
