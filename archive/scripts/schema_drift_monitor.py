#!/usr/bin/env python3
"""
Daily Schema Drift Monitor

This script runs daily in production to:
1. Dump current production schema
2. Compare with repository schema baseline  
3. Alert on any drift detected
4. Save snapshots for historical tracking

Usage:
    python scripts/schema_drift_monitor.py --prod-db-url="postgresql://..." --alert-webhook="..."
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

# Add the project root to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_database_schema_consistency import DatabaseSchemaConsistencyTester


class SchemaDriftMonitor:
    """Production schema drift monitoring and alerting system."""
    
    def __init__(self, prod_db_url: str, baseline_snapshot: str = None, 
                 alert_webhook: str = None, email_config: dict = None):
        self.prod_db_url = prod_db_url
        self.baseline_snapshot = baseline_snapshot or "schema_baseline.json"
        self.alert_webhook = alert_webhook
        self.email_config = email_config or {}
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize tester
        self.tester = DatabaseSchemaConsistencyTester(prod_db_url)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("schema_drift_monitor")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"schema_drift_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_baseline_snapshot(self) -> str:
        """Create a baseline schema snapshot from the current production database."""
        self.logger.info("Creating baseline schema snapshot...")
        
        try:
            snapshot_file = self.tester.save_schema_snapshot(self.baseline_snapshot)
            self.logger.info(f"Baseline snapshot saved to: {snapshot_file}")
            return snapshot_file
        except Exception as e:
            self.logger.error(f"Failed to create baseline snapshot: {str(e)}")
            raise
    
    def run_daily_check(self) -> dict:
        """Run the daily schema drift check."""
        self.logger.info("Starting daily schema drift check...")
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "drift_detected": False,
            "schema_hash": None,
            "differences": None,
            "alerts_sent": [],
            "errors": []
        }
        
        try:
            # Generate current schema snapshot
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_snapshot = f"schema_snapshot_prod_{current_timestamp}.json"
            
            self.logger.info("Generating current production schema snapshot...")
            self.tester.save_schema_snapshot(current_snapshot)
            result["current_snapshot"] = current_snapshot
            
            # Generate schema hash
            schema_hash = self.tester.generate_schema_hash()
            result["schema_hash"] = schema_hash
            self.logger.info(f"Current schema hash: {schema_hash}")
            
            # Compare with baseline if it exists
            if Path(self.baseline_snapshot).exists():
                self.logger.info(f"Comparing with baseline: {self.baseline_snapshot}")
                differences = self.tester.compare_schema_snapshots(
                    self.baseline_snapshot, current_snapshot
                )
                result["differences"] = differences
                
                # Check if drift detected
                if differences["schema_hash_changed"] or differences["summary"]:
                    result["drift_detected"] = True
                    self.logger.warning("Schema drift detected!")
                    self.logger.warning(f"Changes: {differences['summary']}")
                    
                    # Send alerts
                    alert_sent = self._send_drift_alerts(differences, schema_hash)
                    result["alerts_sent"] = alert_sent
                else:
                    self.logger.info("No schema drift detected")
            else:
                self.logger.warning(f"Baseline snapshot not found: {self.baseline_snapshot}")
                self.logger.info("Creating baseline from current production schema...")
                self.create_baseline_snapshot()
                result["baseline_created"] = True
            
            # Run additional integrity checks
            self.logger.info("Running schema consistency checks...")
            consistency_result = self.tester.test_alembic_schema_consistency()
            fk_index_result = self.tester.test_foreign_key_indexes()
            integrity_result = self.tester.test_data_integrity()
            
            result["consistency_checks"] = {
                "alembic_consistency": consistency_result,
                "foreign_key_indexes": fk_index_result,
                "data_integrity": integrity_result
            }
            
            # Check for critical issues
            critical_issues = []
            if not consistency_result["passed"]:
                critical_issues.append("Schema-model mismatch detected")
            if not fk_index_result["passed"]:
                critical_issues.append("Missing foreign key indexes")
            if not integrity_result["passed"]:
                critical_issues.append("Data integrity violations")
            
            if critical_issues:
                result["critical_issues"] = critical_issues
                self.logger.error(f"Critical issues found: {critical_issues}")
                alert_sent = self._send_critical_alerts(critical_issues, result["consistency_checks"])
                result["alerts_sent"].extend(alert_sent)
            
            # Clean up old snapshots (keep only last 30 days)
            self._cleanup_old_snapshots()
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            self.logger.error(f"Schema drift check failed: {str(e)}")
            
            # Send error alert
            error_alert = self._send_error_alert(str(e))
            result["alerts_sent"].append(error_alert)
        
        self.logger.info(f"Schema drift check completed. Status: {result['status']}")
        return result
    
    def _send_drift_alerts(self, differences: dict, schema_hash: str) -> list:
        """Send alerts when schema drift is detected."""
        alerts_sent = []
        
        # Prepare alert message
        alert_data = {
            "title": "🚨 Database Schema Drift Detected",
            "message": f"Schema changes detected in production database",
            "details": {
                "timestamp": datetime.now().isoformat(),
                "schema_hash": schema_hash,
                "changes": differences["summary"],
                "hash_changed": differences["schema_hash_changed"]
            },
            "severity": "warning",
            "service": "greyhound_racing_predictor"
        }
        
        # Send webhook alert
        if self.alert_webhook:
            try:
                webhook_sent = self._send_webhook_alert(alert_data)
                alerts_sent.append(webhook_sent)
            except Exception as e:
                self.logger.error(f"Failed to send webhook alert: {str(e)}")
        
        # Send email alert
        if self.email_config:
            try:
                email_sent = self._send_email_alert(alert_data)
                alerts_sent.append(email_sent)
            except Exception as e:
                self.logger.error(f"Failed to send email alert: {str(e)}")
        
        return alerts_sent
    
    def _send_critical_alerts(self, issues: list, check_results: dict) -> list:
        """Send alerts for critical schema/integrity issues."""
        alerts_sent = []
        
        alert_data = {
            "title": "🔥 Critical Database Issues Detected",
            "message": f"Critical schema or integrity issues found",
            "details": {
                "timestamp": datetime.now().isoformat(),
                "issues": issues,
                "check_results": check_results
            },
            "severity": "critical",
            "service": "greyhound_racing_predictor"
        }
        
        # Send webhook alert
        if self.alert_webhook:
            try:
                webhook_sent = self._send_webhook_alert(alert_data)
                alerts_sent.append(webhook_sent)
            except Exception as e:
                self.logger.error(f"Failed to send critical webhook alert: {str(e)}")
        
        # Send email alert
        if self.email_config:
            try:
                email_sent = self._send_email_alert(alert_data)
                alerts_sent.append(email_sent)
            except Exception as e:
                self.logger.error(f"Failed to send critical email alert: {str(e)}")
        
        return alerts_sent
    
    def _send_error_alert(self, error_message: str) -> dict:
        """Send alert when the monitoring script itself fails."""
        alert_data = {
            "title": "💥 Schema Monitoring Script Failed",
            "message": f"Schema drift monitoring script encountered an error",
            "details": {
                "timestamp": datetime.now().isoformat(),
                "error": error_message
            },
            "severity": "error",
            "service": "greyhound_racing_predictor"
        }
        
        # Try to send at least one alert
        try:
            if self.alert_webhook:
                return self._send_webhook_alert(alert_data)
            elif self.email_config:
                return self._send_email_alert(alert_data)
        except Exception as e:
            self.logger.error(f"Failed to send error alert: {str(e)}")
        
        return {"type": "error", "status": "failed", "error": str(e)}
    
    def _send_webhook_alert(self, alert_data: dict) -> dict:
        """Send webhook alert (Slack, Discord, Teams, etc.)."""
        try:
            response = requests.post(
                self.alert_webhook,
                json=alert_data,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            return {
                "type": "webhook",
                "status": "sent",
                "webhook_url": self.alert_webhook,
                "response_code": response.status_code
            }
        except Exception as e:
            raise Exception(f"Webhook alert failed: {str(e)}")
    
    def _send_email_alert(self, alert_data: dict) -> dict:
        """Send email alert."""
        try:
            smtp_server = self.email_config.get("smtp_server")
            smtp_port = self.email_config.get("smtp_port", 587)
            username = self.email_config.get("username")
            password = self.email_config.get("password")
            from_email = self.email_config.get("from_email", username)
            to_emails = self.email_config.get("to_emails", [])
            
            if not all([smtp_server, username, password, to_emails]):
                raise Exception("Incomplete email configuration")
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = alert_data["title"]
            
            # Email body
            body = f"""
{alert_data['message']}

Details:
{json.dumps(alert_data['details'], indent=2)}

Timestamp: {datetime.now().isoformat()}
Service: {alert_data.get('service', 'Unknown')}
Severity: {alert_data.get('severity', 'Unknown')}
            """
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return {
                "type": "email",
                "status": "sent",
                "to_emails": to_emails,
                "subject": alert_data["title"]
            }
        except Exception as e:
            raise Exception(f"Email alert failed: {str(e)}")
    
    def _cleanup_old_snapshots(self, keep_days: int = 30):
        """Clean up old schema snapshots to save disk space."""
        try:
            current_time = datetime.now()
            snapshots_dir = Path(".")
            
            for snapshot_file in snapshots_dir.glob("schema_snapshot_*.json"):
                if snapshot_file.name == self.baseline_snapshot:
                    continue  # Don't delete baseline
                
                file_age = current_time - datetime.fromtimestamp(snapshot_file.stat().st_mtime)
                if file_age.days > keep_days:
                    snapshot_file.unlink()
                    self.logger.info(f"Cleaned up old snapshot: {snapshot_file}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old snapshots: {str(e)}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Daily Schema Drift Monitor")
    parser.add_argument("--prod-db-url", required=True, 
                       help="Production database URL")
    parser.add_argument("--baseline-snapshot", default="schema_baseline.json",
                       help="Baseline schema snapshot file")
    parser.add_argument("--alert-webhook", 
                       help="Webhook URL for alerts (optional)")
    parser.add_argument("--email-smtp-server", 
                       help="SMTP server for email alerts")
    parser.add_argument("--email-username", 
                       help="Email username")
    parser.add_argument("--email-password", 
                       help="Email password")
    parser.add_argument("--email-to", action="append",
                       help="Email recipients (can be used multiple times)")
    parser.add_argument("--create-baseline", action="store_true",
                       help="Create baseline snapshot and exit")
    parser.add_argument("--output-file", 
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Prepare email configuration
    email_config = None
    if args.email_smtp_server and args.email_username and args.email_password:
        email_config = {
            "smtp_server": args.email_smtp_server,
            "username": args.email_username,
            "password": args.email_password,
            "to_emails": args.email_to or []
        }
    
    # Initialize monitor
    monitor = SchemaDriftMonitor(
        prod_db_url=args.prod_db_url,
        baseline_snapshot=args.baseline_snapshot,
        alert_webhook=args.alert_webhook,
        email_config=email_config
    )
    
    try:
        if args.create_baseline:
            # Just create baseline and exit
            snapshot_file = monitor.create_baseline_snapshot()
            print(f"Baseline snapshot created: {snapshot_file}")
            return 0
        
        # Run daily check
        result = monitor.run_daily_check()
        
        # Save results to file if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        
        # Print summary
        print(f"Schema drift check completed: {result['status']}")
        if result.get("drift_detected"):
            print("⚠️  Schema drift detected!")
        if result.get("critical_issues"):
            print(f"🔥 Critical issues: {result['critical_issues']}")
            return 1
        
        print("✅ Schema monitoring completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Schema monitoring failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
