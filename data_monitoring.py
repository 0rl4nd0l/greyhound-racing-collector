#!/usr/bin/env python3
"""
Data Monitoring and Alerting System
====================================

This script monitors the greyhound racing database for integrity issues and sends alerts
when problems are detected. It can be run regularly via cron job.

Features:
- Real-time integrity monitoring
- Email alerts for critical issues
- Trend analysis and reporting
- Integration with data integrity system

Author: AI Assistant
Date: 2025-01-27
"""

import json
import logging
import os
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from data_integrity_system import DataIntegrityManager


class DataMonitor:
    """Main class for monitoring data integrity and sending alerts"""

    def __init__(
        self,
        db_path: str = "greyhound_racing_data.db",
        config_path: str = "config/monitoring.json",
    ):
        self.db_path = db_path
        self.config_path = config_path
        self.integrity_manager = DataIntegrityManager(db_path)
        self.setup_logging()
        self.load_config()
        self.monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "alerts": [],
            "metrics": {},
            "trends": {},
            "recommendations": [],
        }

    def setup_logging(self):
        """Setup logging for monitoring operations"""
        os.makedirs("logs", exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - MONITOR - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/data_monitoring.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        """Load monitoring configuration"""
        os.makedirs("config", exist_ok=True)

        default_config = {
            "alert_thresholds": {
                "max_dog_day_violations": 10,
                "max_invalid_box_numbers": 50,
                "max_duplicate_records": 5,
                "min_data_quality_score": 95.0,
            },
            "email_settings": {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "sender_email": "monitoring@greyhound-racing.local",
                "recipient_emails": ["admin@greyhound-racing.local"],
                "use_tls": True,
                "username": "",
                "password": "",
            },
            "monitoring_schedule": {
                "check_interval_minutes": 60,
                "daily_report_time": "08:00",
                "weekly_report_day": "monday",
            },
            "trend_analysis": {"lookback_days": 30, "min_data_points": 5},
        }

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in self.config:
                            self.config[key] = value
            except Exception as e:
                self.logger.warning(f"Error loading config, using defaults: {e}")
                self.config = default_config
        else:
            self.config = default_config
            # Save default config
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Created default config file: {self.config_path}")

    def check_integrity_violations(self) -> Dict:
        """Check for current integrity violations"""
        with self.integrity_manager:
            integrity_report = (
                self.integrity_manager.run_comprehensive_integrity_check()
            )

        violations = {
            "dog_day_violations": 0,
            "invalid_box_numbers": 0,
            "duplicate_records": 0,
            "data_quality_issues": 0,
        }

        # Parse integrity report for violations
        for issue in integrity_report.get("issues_found", []):
            if "dog-day rule violations" in issue:
                violations["dog_day_violations"] += int(issue.split()[1])
            elif "invalid box numbers" in issue:
                violations["invalid_box_numbers"] += int(issue.split()[1])
            elif "duplicate" in issue.lower():
                violations["duplicate_records"] += 1
            else:
                violations["data_quality_issues"] += 1

        return violations

    def calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get total record counts
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            total_dog_races = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
            total_expert_data = cursor.fetchone()[0]

            if total_races == 0:
                return 0.0

            # Calculate quality metrics
            quality_metrics = {"completeness": 0.0, "consistency": 0.0, "validity": 0.0}

            # Completeness: Check for NULL values in critical fields
            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE race_id IS NOT NULL AND venue IS NOT NULL AND race_date IS NOT NULL
            """
            )
            complete_races = cursor.fetchone()[0]
            quality_metrics["completeness"] = (complete_races / total_races) * 100

            # Consistency: Check for data consistency
            cursor.execute(
                """
                SELECT COUNT(*) FROM dog_race_data 
                WHERE box_number BETWEEN 1 AND 8 AND finish_position BETWEEN 1 AND 8
            """
            )
            consistent_dog_data = cursor.fetchone()[0]
            if total_dog_races > 0:
                quality_metrics["consistency"] = (
                    consistent_dog_data / total_dog_races
                ) * 100
            else:
                quality_metrics["consistency"] = 100.0

            # Validity: Check for valid data formats and ranges
            cursor.execute(
                """
                SELECT COUNT(*) FROM enhanced_expert_data 
                WHERE position IS NULL OR (position BETWEEN 1 AND 8)
            """
            )
            valid_positions = cursor.fetchone()[0]
            if total_expert_data > 0:
                quality_metrics["validity"] = (
                    valid_positions / total_expert_data
                ) * 100
            else:
                quality_metrics["validity"] = 100.0

            # Overall score is weighted average
            overall_score = (
                quality_metrics["completeness"] * 0.4
                + quality_metrics["consistency"] * 0.4
                + quality_metrics["validity"] * 0.2
            )

            return round(overall_score, 2)

        except sqlite3.Error as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.0
        finally:
            conn.close()

    def analyze_trends(self) -> Dict:
        """Analyze data trends over time"""
        trends = {}

        # Check if we have historical monitoring data
        monitoring_logs_dir = "logs/monitoring_history"
        os.makedirs(monitoring_logs_dir, exist_ok=True)

        history_file = os.path.join(monitoring_logs_dir, "monitoring_history.json")

        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)

                # Analyze recent trends
                recent_data = []
                cutoff_date = datetime.now() - timedelta(
                    days=self.config["trend_analysis"]["lookback_days"]
                )

                for entry in history:
                    entry_date = datetime.fromisoformat(entry["timestamp"])
                    if entry_date >= cutoff_date:
                        recent_data.append(entry)

                if len(recent_data) >= self.config["trend_analysis"]["min_data_points"]:
                    # Calculate trends
                    quality_scores = [
                        entry.get("data_quality_score", 0) for entry in recent_data
                    ]
                    violation_counts = [
                        len(entry.get("violations", {})) for entry in recent_data
                    ]

                    trends["data_quality_trend"] = (
                        "improving"
                        if quality_scores[-1] > quality_scores[0]
                        else "declining"
                    )
                    trends["violation_trend"] = (
                        "improving"
                        if violation_counts[-1] < violation_counts[0]
                        else "increasing"
                    )
                    trends["avg_quality_score"] = sum(quality_scores) / len(
                        quality_scores
                    )
                    trends["avg_violations"] = sum(violation_counts) / len(
                        violation_counts
                    )

            except Exception as e:
                self.logger.warning(f"Error analyzing trends: {e}")

        return trends

    def save_monitoring_history(self, current_results: Dict):
        """Save current monitoring results to history"""
        monitoring_logs_dir = "logs/monitoring_history"
        os.makedirs(monitoring_logs_dir, exist_ok=True)

        history_file = os.path.join(monitoring_logs_dir, "monitoring_history.json")

        # Load existing history
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load monitoring history: {e}")

        # Add current results
        history.append(current_results)

        # Keep only recent history (last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        history = [
            entry
            for entry in history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]

        # Save updated history
        try:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save monitoring history: {e}")

    def generate_alerts(self, violations: Dict, quality_score: float) -> List[Dict]:
        """Generate alerts based on thresholds"""
        alerts = []
        thresholds = self.config["alert_thresholds"]

        # Check each threshold
        if violations["dog_day_violations"] > thresholds["max_dog_day_violations"]:
            alerts.append(
                {
                    "type": "CRITICAL",
                    "category": "dog_day_violations",
                    "message": f"Excessive dog-day violations detected: {violations['dog_day_violations']} (threshold: {thresholds['max_dog_day_violations']})",
                    "count": violations["dog_day_violations"],
                    "action_required": "Run deduplication script immediately",
                }
            )

        if violations["invalid_box_numbers"] > thresholds["max_invalid_box_numbers"]:
            alerts.append(
                {
                    "type": "WARNING",
                    "category": "invalid_box_numbers",
                    "message": f"High number of invalid box numbers: {violations['invalid_box_numbers']} (threshold: {thresholds['max_invalid_box_numbers']})",
                    "count": violations["invalid_box_numbers"],
                    "action_required": "Review data ingestion process",
                }
            )

        if violations["duplicate_records"] > thresholds["max_duplicate_records"]:
            alerts.append(
                {
                    "type": "WARNING",
                    "category": "duplicate_records",
                    "message": f"Duplicate records detected: {violations['duplicate_records']} groups (threshold: {thresholds['max_duplicate_records']})",
                    "count": violations["duplicate_records"],
                    "action_required": "Run automated deduplication",
                }
            )

        if quality_score < thresholds["min_data_quality_score"]:
            alerts.append(
                {
                    "type": "CRITICAL",
                    "category": "data_quality",
                    "message": f"Data quality score below threshold: {quality_score}% (threshold: {thresholds['min_data_quality_score']}%)",
                    "score": quality_score,
                    "action_required": "Investigate data quality issues",
                }
            )

        return alerts

    def send_email_alerts(self, alerts: List[Dict]):
        """Send email alerts for critical issues"""
        if not alerts:
            return

        email_config = self.config["email_settings"]

        # Create email content
        subject = f"Greyhound Racing Data Alert - {len(alerts)} issues detected"

        body = f"""
Data Integrity Alert Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{len(alerts)} issue(s) detected:

"""

        for i, alert in enumerate(alerts, 1):
            body += f"{i}. [{alert['type']}] {alert['category'].upper()}\n"
            body += f"   {alert['message']}\n"
            body += f"   Action Required: {alert['action_required']}\n\n"

        body += """
Please review the system logs and take appropriate action.

System: Greyhound Racing Data Monitor
"""

        try:
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = email_config["sender_email"]
            msg["To"] = ", ".join(email_config["recipient_emails"])
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(
                email_config["smtp_server"], email_config["smtp_port"]
            )

            if email_config["use_tls"]:
                server.starttls()

            if email_config["username"]:
                server.login(email_config["username"], email_config["password"])

            server.send_message(msg)
            server.quit()

            self.logger.info(
                f"Email alert sent to {len(email_config['recipient_emails'])} recipients"
            )

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def generate_monitoring_chart(self, trends: Dict) -> str:
        """Generate monitoring trend chart"""
        try:
            monitoring_logs_dir = "logs/monitoring_history"
            history_file = os.path.join(monitoring_logs_dir, "monitoring_history.json")

            if not os.path.exists(history_file):
                return None

            with open(history_file, "r") as f:
                history = json.load(f)

            if len(history) < 2:
                return None

            # Extract data for plotting
            dates = [datetime.fromisoformat(entry["timestamp"]) for entry in history]
            quality_scores = [entry.get("data_quality_score", 0) for entry in history]

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates, quality_scores, marker="o", linewidth=2, markersize=6)
            plt.title("Data Quality Score Trend", fontsize=16, fontweight="bold")
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Quality Score (%)", fontsize=12)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save chart
            chart_path = f"reports/monitoring_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            os.makedirs("reports", exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            return chart_path

        except Exception as e:
            self.logger.error(f"Error generating monitoring chart: {e}")
            return None

    def run_monitoring_check(self) -> Dict:
        """Run complete monitoring check"""
        self.logger.info("Starting data monitoring check")

        # Check for integrity violations
        violations = self.check_integrity_violations()

        # Calculate data quality score
        quality_score = self.calculate_data_quality_score()

        # Analyze trends
        trends = self.analyze_trends()

        # Generate alerts
        alerts = self.generate_alerts(violations, quality_score)

        # Compile results
        results = {
            "timestamp": datetime.now().isoformat(),
            "violations": violations,
            "data_quality_score": quality_score,
            "trends": trends,
            "alerts": alerts,
            "metrics": {
                "total_violations": sum(violations.values()),
                "alert_count": len(alerts),
                "critical_alerts": len([a for a in alerts if a["type"] == "CRITICAL"]),
            },
        }

        # Save to history
        self.save_monitoring_history(results)

        # Send alerts if any critical issues
        critical_alerts = [a for a in alerts if a["type"] == "CRITICAL"]
        if critical_alerts:
            self.send_email_alerts(critical_alerts)

        # Generate chart
        chart_path = self.generate_monitoring_chart(trends)
        if chart_path:
            results["chart_path"] = chart_path

        self.logger.info(
            f"Monitoring check complete: {len(alerts)} alerts, quality score: {quality_score}%"
        )

        return results


def main():
    """Main function for running monitoring checks"""
    print("=== Data Monitoring System ===\\n")

    monitor = DataMonitor()

    try:
        # Run monitoring check
        results = monitor.run_monitoring_check()

        print(f"Monitoring Check Results ({results['timestamp']})")
        print("=" * 50)

        # Display quality score
        quality_score = results["data_quality_score"]
        print(f"Data Quality Score: {quality_score}%")

        if quality_score >= 95:
            print("✅ Excellent data quality")
        elif quality_score >= 85:
            print("⚠️  Good data quality with minor issues")
        else:
            print("❌ Poor data quality - attention required")

        # Display violations
        violations = results["violations"]
        total_violations = sum(violations.values())

        print(f"\\nViolations Detected: {total_violations}")
        if total_violations > 0:
            for violation_type, count in violations.items():
                if count > 0:
                    print(f"  - {violation_type}: {count}")

        # Display alerts
        alerts = results["alerts"]
        if alerts:
            print(f"\\n🚨 ALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"  [{alert['type']}] {alert['message']}")
                print(f"      Action: {alert['action_required']}")
        else:
            print("\\n✅ No alerts generated")

        # Display trends
        trends = results["trends"]
        if trends:
            print(f"\\n📈 TRENDS:")
            for trend_name, trend_value in trends.items():
                print(f"  - {trend_name}: {trend_value}")

        # Display chart path
        if "chart_path" in results:
            print(f"\\n📊 Trend chart generated: {results['chart_path']}")

        return 0

    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
