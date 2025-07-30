#!/usr/bin/env python3
"""
Greyhound Racing Automation Scheduler
Orchestrates daily data collection, processing, analysis, and maintenance tasks
"""

import os
import sys
import time
import json
import logging
import schedule
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

class AutomationScheduler:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.logs_dir = self.base_dir / "logs" / "automation"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / f"automation_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Status tracking
        self.status = {
            'last_run': {},
            'success_count': 0,
            'error_count': 0,
            'running': False,
            'start_time': None
        }
        
        self.running = True
        
    def run_task(self, task_name: str, command: List[str], timeout: int = 300) -> bool:
        """Execute a task with timeout and error handling"""
        self.logger.info(f"🚀 Starting task: {task_name}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"✅ {task_name} completed successfully in {duration:.1f}s")
                self.status['last_run'][task_name] = {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'duration': duration,
                    'output': result.stdout[-500:] if result.stdout else ""
                }
                self.status['success_count'] += 1
                return True
            else:
                self.logger.error(f"❌ {task_name} failed with code {result.returncode}")
                self.logger.error(f"Error: {result.stderr}")
                self.status['last_run'][task_name] = {
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'duration': duration,
                    'error': result.stderr[-500:] if result.stderr else "Unknown error"
                }
                self.status['error_count'] += 1
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏱️ {task_name} timed out after {timeout}s")
            self.status['last_run'][task_name] = {
                'status': 'timeout',
                'timestamp': datetime.now().isoformat(),
                'duration': timeout,
                'error': f"Task timed out after {timeout} seconds"
            }
            self.status['error_count'] += 1
            return False
            
        except Exception as e:
            self.logger.error(f"💥 {task_name} crashed: {str(e)}")
            self.status['last_run'][task_name] = {
                'status': 'crashed',
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - start_time,
                'error': str(e)
            }
            self.status['error_count'] += 1
            return False

    def collect_upcoming_races(self):
        """Collect upcoming race data"""
        return self.run_task(
            "Collect Upcoming Races",
            [sys.executable, "upcoming_race_browser.py"],
            timeout=600  # 10 minutes
        )

    def process_historical_races(self):
        """Process any unprocessed historical races"""
        return self.run_task(
            "Process Historical Races",
            [sys.executable, "enhanced_comprehensive_processor.py"],
            timeout=1800  # 30 minutes
        )

    def update_sportsbet_odds(self):
        """Update live odds from Sportsbet"""
        return self.run_task(
            "Update Sportsbet Odds",
            [sys.executable, "odds_monitor.py", "--single-run"],
            timeout=300  # 5 minutes
        )

    def run_predictions(self):
        """Run predictions on upcoming races using batch method"""
        upcoming_dir = self.base_dir / "upcoming_races"
        
        if not upcoming_dir.exists():
            self.logger.info("📝 No upcoming_races directory found")
            return True
        
        race_files = list(upcoming_dir.glob("*.csv"))
        # Filter out README files
        race_files = [f for f in race_files if f.name != "README.md"]
        
        if not race_files:
            self.logger.info("📝 No upcoming races to predict")
            return True
        
        self.logger.info(f"🎯 Found {len(race_files)} races to predict")
        
        # Use comprehensive prediction pipeline with batch method
        try:
            # Try to use the comprehensive pipeline batch method first
            pipeline_script = "comprehensive_prediction_pipeline.py"
            if (self.base_dir / pipeline_script).exists():
                return self.run_task(
                    "Batch Predictions (Comprehensive Pipeline)",
                    [sys.executable, pipeline_script, "all"],
                    timeout=900  # 15 minutes for batch processing
                )
        except Exception as e:
            self.logger.warning(f"Comprehensive pipeline failed: {e}")
        
        # Fallback to individual predictions using upcoming race predictor
        predict_script = "upcoming_race_predictor.py"
        if not (self.base_dir / predict_script).exists():
            self.logger.error("No prediction scripts found")
            return False
        
        # Run predictions on each race file (limited fallback)
        success_count = 0
        for race_file in race_files[:5]:  # Limit to 5 most recent races
            self.logger.info(f"🏁 Predicting: {race_file.name}")
            
            success = self.run_task(
                f"Predict {race_file.name}",
                [sys.executable, predict_script, str(race_file)],
                timeout=300  # 5 minutes per race
            )
            
            if success:
                success_count += 1
                
            # Brief pause between predictions
            time.sleep(5)
        
        self.logger.info(f"🏆 Predictions completed: {success_count}/{len(race_files[:5])} successful")
        return success_count > 0

    def run_ml_backtesting(self):
        """Run ML backtesting and model updates"""
        return self.run_task(
            "ML Backtesting",
            [sys.executable, "ml_backtesting_trainer.py"],
            timeout=1200  # 20 minutes
        )

    def generate_reports(self):
        """Generate daily analysis reports"""
        return self.run_task(
            "Generate Reports",
            [sys.executable, "-c", """
import sys
sys.path.append('.')
from app import app
with app.app_context():
    from enhanced_race_analyzer import EnhancedRaceAnalyzer
    analyzer = EnhancedRaceAnalyzer()
    analyzer.generate_comprehensive_report()
"""],
            timeout=600  # 10 minutes
        )

    def cleanup_old_files(self):
        """Clean up only temporary files - keep all valuable data"""
        try:
            # Only clean truly temporary/junk files - keep all logs, predictions, etc.
            temp_cleaned = 0
            temp_patterns = [
                '*.tmp', '*.temp', '*~', '.DS_Store', '*.pyc', '__pycache__',
                '*.swp', '*.swo', '.pytest_cache', '*.lock'
            ]
            
            for pattern in temp_patterns:
                for temp_file in self.base_dir.rglob(pattern):
                    if temp_file.is_file():
                        temp_file.unlink()
                        temp_cleaned += 1
                    elif temp_file.is_dir() and pattern == '__pycache__':
                        import shutil
                        shutil.rmtree(temp_file)
                        temp_cleaned += 1
            
            # Get storage usage summary for monitoring
            total_size = sum(f.stat().st_size for f in self.base_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            # Count files by type for insight
            file_counts = {
                'logs': len(list(self.base_dir.rglob('*.log'))),
                'predictions': len(list((self.base_dir / 'predictions').glob('*.json'))) if (self.base_dir / 'predictions').exists() else 0,
                'csv_files': len(list(self.base_dir.rglob('*.csv'))),
                'databases': len(list(self.base_dir.rglob('*.db'))),
                'backups': len(list((self.base_dir / 'backups').glob('*.db'))) if (self.base_dir / 'backups').exists() else 0
            }
            
            self.logger.info(f"🧹 Cleanup completed - removed {temp_cleaned} temporary files")
            self.logger.info(f"💾 Total storage: {total_size_mb:.1f} MB")
            self.logger.info(f"📊 Data files: {file_counts['logs']} logs, {file_counts['predictions']} predictions, {file_counts['csv_files']} CSVs, {file_counts['databases']} DBs, {file_counts['backups']} backups")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {str(e)}")
            return False

    def backup_database(self):
        """Create daily database backup with intelligent retention"""
        try:
            db_file = self.base_dir / "greyhound_racing_data.db"
            if not db_file.exists():
                self.logger.warning("⚠️ Database file not found for backup")
                return False
                
            backup_dir = self.base_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            current_date = datetime.now()
            backup_file = backup_dir / f"greyhound_racing_data_backup_{current_date.strftime('%Y%m%d')}.db"
            
            # Skip if backup already exists for today
            if backup_file.exists():
                self.logger.info(f"💾 Daily backup already exists: {backup_file.name}")
                return True
            
            # Create backup using sqlite3
            source_conn = sqlite3.connect(str(db_file))
            backup_conn = sqlite3.connect(str(backup_file))
            source_conn.backup(backup_conn)
            source_conn.close()
            backup_conn.close()
            
            # Smart retention policy:
            # - Keep daily backups for last 30 days
            # - Keep weekly backups (Sundays) for last 12 weeks  
            # - Keep monthly backups (1st of month) for last 12 months
            # - Keep yearly backups (Jan 1st) forever
            
            backup_files = sorted(backup_dir.glob("greyhound_racing_data_backup_*.db"))
            keep_files = set()
            
            for backup_path in backup_files:
                try:
                    # Extract date from filename
                    date_str = backup_path.name.split('_')[-1].replace('.db', '')
                    backup_date = datetime.strptime(date_str, '%Y%m%d')
                    days_old = (current_date - backup_date).days
                    
                    # Keep daily for 30 days
                    if days_old <= 30:
                        keep_files.add(backup_path)
                    # Keep weekly (Sundays) for 12 weeks
                    elif days_old <= 84 and backup_date.weekday() == 6:  # Sunday = 6
                        keep_files.add(backup_path)
                    # Keep monthly (1st of month) for 12 months
                    elif days_old <= 365 and backup_date.day == 1:
                        keep_files.add(backup_path)
                    # Keep yearly (Jan 1st) forever
                    elif backup_date.month == 1 and backup_date.day == 1:
                        keep_files.add(backup_path)
                        
                except (ValueError, IndexError):
                    # Keep files that don't match expected format
                    keep_files.add(backup_path)
            
            # Remove backups not in keep list
            removed_count = 0
            for backup_path in backup_files:
                if backup_path not in keep_files:
                    backup_path.unlink()
                    removed_count += 1
            
            self.logger.info(f"💾 Database backed up to {backup_file.name}")
            if removed_count > 0:
                self.logger.info(f"🗂️ Cleaned {removed_count} old backups (kept {len(keep_files)} backups)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database backup failed: {str(e)}")
            return False

    def data_integrity_check(self):
        """Run data integrity checks"""
        return self.run_task(
            "Data Integrity Check",
            [sys.executable, "data_integrity_checker.py"],
            timeout=300  # 5 minutes
        )

    def save_status(self):
        """Save current status to file"""
        status_file = self.logs_dir / "automation_status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump(self.status, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save status: {e}")

    def morning_routine(self):
        """Complete morning data collection and processing routine"""
        self.logger.info("🌅 Starting morning routine")
        
        tasks = [
            ("Database Backup", self.backup_database),
            ("Collect Upcoming Races", self.collect_upcoming_races),
            ("Update Odds", self.update_sportsbet_odds),
            ("Run Predictions", self.run_predictions),
            ("Data Integrity Check", self.data_integrity_check)
        ]
        
        success_count = 0
        for task_name, task_func in tasks:
            if task_func():
                success_count += 1
            time.sleep(10)  # Brief pause between tasks
        
        self.logger.info(f"🌅 Morning routine completed: {success_count}/{len(tasks)} tasks successful")
        self.save_status()

    def afternoon_routine(self):
        """Afternoon processing and analysis routine"""
        self.logger.info("🌞 Starting afternoon routine")
        
        tasks = [
            ("Process Historical Races", self.process_historical_races),
            ("Update Odds", self.update_sportsbet_odds),
            ("Run Predictions", self.run_predictions),
            ("Generate Reports", self.generate_reports)
        ]
        
        success_count = 0
        for task_name, task_func in tasks:
            if task_func():
                success_count += 1
            time.sleep(10)
        
        self.logger.info(f"🌞 Afternoon routine completed: {success_count}/{len(tasks)} tasks successful")
        self.save_status()

    def evening_routine(self):
        """Evening analysis and maintenance routine"""
        self.logger.info("🌙 Starting evening routine")
        
        tasks = [
            ("Process Historical Races", self.process_historical_races),
            ("ML Backtesting", self.run_ml_backtesting),
            ("Generate Reports", self.generate_reports),
            ("Cleanup Old Files", self.cleanup_old_files),
            ("Data Integrity Check", self.data_integrity_check)
        ]
        
        success_count = 0
        for task_name, task_func in tasks:
            if task_func():
                success_count += 1
            time.sleep(10)
        
        self.logger.info(f"🌙 Evening routine completed: {success_count}/{len(tasks)} tasks successful")
        self.save_status()

    def setup_schedule(self):
        """Setup the daily schedule"""
        # Morning routine - collect fresh data
        schedule.every().day.at("07:00").do(self.morning_routine)
        
        # Afternoon routine - process and analyze
        schedule.every().day.at("14:00").do(self.afternoon_routine)
        
        # Evening routine - comprehensive analysis and maintenance
        schedule.every().day.at("20:00").do(self.evening_routine)
        
        # Quick odds updates every 2 hours during race times (9 AM - 10 PM)
        for hour in range(9, 23, 2):
            schedule.every().day.at(f"{hour:02d}:00").do(self.update_sportsbet_odds)
        
        # Weekly comprehensive ML training (Sundays at 22:00)
        schedule.every().sunday.at("22:00").do(self.run_ml_backtesting)
        
        self.logger.info("📅 Automation schedule configured:")
        self.logger.info("  • 07:00 - Morning routine (collect, odds, predict)")
        self.logger.info("  • 14:00 - Afternoon routine (process, analyze)")
        self.logger.info("  • 20:00 - Evening routine (ML, reports, maintenance)")
        self.logger.info("  • Every 2h (9-22) - Odds updates")
        self.logger.info("  • Sundays 22:00 - Weekly ML training")

    def run_scheduler(self):
        """Main scheduler loop"""
        self.status['running'] = True
        self.status['start_time'] = datetime.now().isoformat()
        
        self.logger.info("🤖 Greyhound Racing Automation Started")
        self.setup_schedule()
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Automation stopped by user")
        except Exception as e:
            self.logger.error(f"💥 Scheduler crashed: {str(e)}")
        finally:
            self.status['running'] = False
            self.save_status()
            self.logger.info("🛑 Automation scheduler stopped")

    def run_single_task(self, task_name: str):
        """Run a single task for testing"""
        task_map = {
            'morning': self.morning_routine,
            'afternoon': self.afternoon_routine,
            'evening': self.evening_routine,
            'collect': self.collect_upcoming_races,
            'process': self.process_historical_races,
            'odds': self.update_sportsbet_odds,
            'predict': self.run_predictions,
            'ml': self.run_ml_backtesting,
            'reports': self.generate_reports,
            'backup': self.backup_database,
            'cleanup': self.cleanup_old_files,
            'integrity': self.data_integrity_check
        }
        
        if task_name in task_map:
            self.logger.info(f"🎯 Running single task: {task_name}")
            task_map[task_name]()
            self.save_status()
        else:
            self.logger.error(f"❌ Unknown task: {task_name}")
            self.logger.info(f"Available tasks: {', '.join(task_map.keys())}")

def main():
    scheduler = AutomationScheduler()
    
    if len(sys.argv) > 1:
        # Run single task
        task = sys.argv[1]
        scheduler.run_single_task(task)
    else:
        # Run full scheduler
        scheduler.run_scheduler()

if __name__ == "__main__":
    main()
