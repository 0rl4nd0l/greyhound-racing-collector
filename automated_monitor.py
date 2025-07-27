#!/usr/bin/env python3
"""
Automated Monitoring System for Greyhound Racing Value Bets
==========================================================

This script runs continuous monitoring of live odds vs ML predictions
to identify and alert on value betting opportunities.

Usage:
    python automated_monitor.py --config config.json
    python automated_monitor.py --urls "url1,url2,url3" --interval 15

Author: Orlando Lee
Date: July 27, 2025
"""

import argparse
import json
import time
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict

from integrated_odds_ml_system import IntegratedOddsMLSystem

class AutomatedMonitor:
    """Automated monitoring service for value betting opportunities"""
    
    def __init__(self, config_file: str = None):
        self.system = IntegratedOddsMLSystem()
        self.config = self.load_config(config_file)
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "race_urls": [
                "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/sale/race-1-9443604"
            ],
            "monitoring_interval_minutes": 15,
            "value_threshold_percentage": 5.0,
            "confidence_threshold": 0.6,
            "max_stake_percentage": 5.0,
            "alert_thresholds": {
                "high_value": 10.0,
                "medium_value": 7.5,
                "low_value": 5.0
            },
            "notifications": {
                "console": True,
                "file": True,
                "email": False,
                "slack": False
            },
            "database_cleanup_days": 30,
            "max_daily_bets": 10
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
                    self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {e}")
                self.logger.info("Using default configuration")
        
        return default_config
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def get_current_race_urls(self) -> List[str]:
        """
        Get current live race URLs. In production, this would scrape
        the main racing page to find current races automatically.
        """
        # For now, return configured URLs
        # TODO: Add automatic race discovery
        return self.config["race_urls"]
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        try:
            self.logger.info("🔄 Starting monitoring cycle...")
            
            # Get current race URLs
            race_urls = self.get_current_race_urls()
            
            if not race_urls:
                self.logger.warning("No race URLs available for monitoring")
                return
            
            # Run complete analysis
            results = self.system.run_complete_analysis(race_urls)
            
            # Process results
            self.process_results(results)
            
            # Log cycle summary
            summary = (
                f"Cycle complete: {results['races_analyzed']} races, "
                f"{results['value_opportunities']} opportunities, "
                f"{len(results.get('errors', []))} errors"
            )
            self.logger.info(summary)
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def process_results(self, results: Dict):
        """Process and act on monitoring results"""
        if results['value_opportunities'] > 0:
            self.logger.info(f"🎯 Found {results['value_opportunities']} value betting opportunities!")
            
            # Process top bets
            for i, bet in enumerate(results.get('top_bets', []), 1):
                value_pct = bet.get('value_percentage', 0)
                
                # Determine alert level
                if value_pct >= self.config['alert_thresholds']['high_value']:
                    alert_level = "HIGH"
                    emoji = "🚨"
                elif value_pct >= self.config['alert_thresholds']['medium_value']:
                    alert_level = "MEDIUM"
                    emoji = "⚠️"
                else:
                    alert_level = "LOW"
                    emoji = "📊"
                
                # Log the opportunity
                alert_msg = (
                    f"{emoji} {alert_level} VALUE BET #{i}\n"
                    f"   Dog: {bet.get('dog_name', 'Unknown')}\n"
                    f"   Venue: {bet.get('venue', 'Unknown')}\n"
                    f"   Value: {value_pct:.1f}%\n"
                    f"   Odds: {bet.get('odds', 0):.2f}\n"
                    f"   Confidence: {bet.get('confidence_score', 0):.3f}\n"
                    f"   Recommended Stake: {bet.get('recommended_stake', 0):.2f}%"
                )
                self.logger.info(alert_msg)
                
                # Send notifications based on configuration
                self.send_notifications(alert_msg, alert_level)
        
        # Log any errors
        if results.get('errors'):
            for error in results['errors']:
                self.logger.error(f"Analysis error: {error}")
    
    def send_notifications(self, message: str, alert_level: str):
        """Send notifications based on configuration"""
        notifications = self.config.get('notifications', {})
        
        # Console notification (already logged)
        if notifications.get('console', True):
            pass  # Already logged above
        
        # File notification
        if notifications.get('file', True):
            alert_file = Path('value_bet_alerts.log')
            with open(alert_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {alert_level}\n{message}\n\n")
        
        # Email notification (placeholder)
        if notifications.get('email', False):
            self.send_email_notification(message, alert_level)
        
        # Slack notification (placeholder)
        if notifications.get('slack', False):
            self.send_slack_notification(message, alert_level)
    
    def send_email_notification(self, message: str, alert_level: str):
        """Send email notification (implement with your email service)"""
        # TODO: Implement email notifications
        # Example using smtplib:
        # import smtplib
        # from email.mime.text import MIMEText
        self.logger.info("Email notification would be sent here")
    
    def send_slack_notification(self, message: str, alert_level: str):
        """Send Slack notification (implement with your Slack webhook)"""
        # TODO: Implement Slack notifications
        # Example using requests to Slack webhook:
        # import requests
        # payload = {"text": message}
        # requests.post(SLACK_WEBHOOK_URL, json=payload)
        self.logger.info("Slack notification would be sent here")
    
    def cleanup_old_data(self):
        """Clean up old data from database"""
        try:
            cleanup_days = self.config.get('database_cleanup_days', 30)
            
            import sqlite3
            conn = sqlite3.connect(self.system.db_path)
            cursor = conn.cursor()
            
            # Clean up old odds data
            cursor.execute('''
                DELETE FROM live_odds_enhanced 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(cleanup_days))
            
            # Clean up old opportunities
            cursor.execute('''
                DELETE FROM value_opportunities 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(cleanup_days))
            
            # Clean up old movements
            cursor.execute('''
                DELETE FROM odds_movements 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(cleanup_days))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up data older than {cleanup_days} days")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
    
    def generate_daily_report(self):
        """Generate and log daily performance report"""
        try:
            report = self.system.get_performance_report(days=1)
            
            report_msg = (
                f"📈 DAILY REPORT - {datetime.now().strftime('%Y-%m-%d')}\n"
                f"   Total Opportunities: {report['total_opportunities']}\n"
                f"   Average Value: {report['avg_value_percentage']:.1f}%\n"
                f"   High Value Bets: {report['high_value_bets']}\n"
                f"   Sportsbooks Monitored: {report['sportsbooks_monitored']}"
            )
            
            self.logger.info(report_msg)
            
            # Save to daily report file
            report_file = Path(f"daily_reports/report_{datetime.now().strftime('%Y%m%d')}.json")
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    def start(self):
        """Start automated monitoring"""
        self.running = True
        self.logger.info("🚀 Starting automated monitoring system...")
        
        # Log configuration
        self.logger.info(f"   Monitoring interval: {self.config['monitoring_interval_minutes']} minutes")
        self.logger.info(f"   Value threshold: {self.config['value_threshold_percentage']}%")
        self.logger.info(f"   Monitoring {len(self.config['race_urls'])} race URLs")
        
        # Set system thresholds
        self.system.value_threshold = self.config['value_threshold_percentage'] / 100
        
        last_cleanup = datetime.now()
        last_daily_report = datetime.now().date()
        
        try:
            while self.running:
                # Run monitoring cycle
                self.run_monitoring_cycle()
                
                # Daily cleanup (once per day)
                if (datetime.now() - last_cleanup).days >= 1:
                    self.cleanup_old_data()
                    last_cleanup = datetime.now()
                
                # Daily report (once per day)
                if datetime.now().date() > last_daily_report:
                    self.generate_daily_report()
                    last_daily_report = datetime.now().date()
                
                # Wait for next cycle
                interval_seconds = self.config['monitoring_interval_minutes'] * 60
                self.logger.info(f"⏱️  Waiting {self.config['monitoring_interval_minutes']} minutes until next cycle...")
                
                # Sleep in small chunks to allow for graceful shutdown
                for _ in range(interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop automated monitoring"""
        self.running = False
        self.system.stop_automated_monitoring()
        self.logger.info("✅ Automated monitoring stopped")

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "race_urls": [
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/sale/race-1-9443604",
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/bendigo/race-2-9443605"
        ],
        "monitoring_interval_minutes": 15,
        "value_threshold_percentage": 5.0,
        "confidence_threshold": 0.6,
        "max_stake_percentage": 5.0,
        "alert_thresholds": {
            "high_value": 10.0,
            "medium_value": 7.5,
            "low_value": 5.0
        },
        "notifications": {
            "console": True,
            "file": True,
            "email": False,
            "slack": False
        },
        "database_cleanup_days": 30,
        "max_daily_bets": 10
    }
    
    with open('monitor_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created sample configuration file: monitor_config.json")
    print("   Edit this file to customize your monitoring settings")

def main():
    parser = argparse.ArgumentParser(description="Automated Greyhound Racing Value Bet Monitor")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--urls', help='Comma-separated list of race URLs')
    parser.add_argument('--interval', type=int, help='Monitoring interval in minutes')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    parser.add_argument('--test', action='store_true', help='Run a single test cycle')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Initialize monitor
    monitor = AutomatedMonitor(args.config)
    
    # Override config with command line arguments
    if args.urls:
        monitor.config['race_urls'] = [url.strip() for url in args.urls.split(',')]
    
    if args.interval:
        monitor.config['monitoring_interval_minutes'] = args.interval
    
    if args.test:
        print("🧪 Running single test cycle...")
        monitor.run_monitoring_cycle()
        print("✅ Test cycle complete")
    else:
        # Start continuous monitoring
        monitor.start()

if __name__ == "__main__":
    main()
