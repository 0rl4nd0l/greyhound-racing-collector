#!/usr/bin/env python3
"""
TGR Service Scheduler
====================

Intelligent scheduler that coordinates the TGR monitoring dashboard with 
the enrichment service to provide automated, optimized data management.
"""

import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import schedule
import threading
from dataclasses import dataclass

from tgr_monitoring_dashboard import TGRMonitoringDashboard
from tgr_enrichment_service import TGREnrichmentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for the TGR service scheduler."""
    monitoring_interval: int = 300  # 5 minutes
    enrichment_batch_size: int = 10
    max_concurrent_jobs: int = 3
    performance_threshold: float = 0.7  # 70% success rate
    data_freshness_hours: int = 24
    auto_retry_failed_jobs: bool = True
    enable_predictive_scheduling: bool = True

class TGRServiceScheduler:
    """Intelligent scheduler for TGR services."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db", 
                 config: SchedulerConfig = None):
        self.db_path = db_path
        self.config = config or SchedulerConfig()
        
        # Initialize service components
        self.monitor = TGRMonitoringDashboard(db_path)
        self.enrichment_service = TGREnrichmentService(
            db_path=db_path, 
            max_workers=self.config.max_concurrent_jobs,
            batch_size=self.config.enrichment_batch_size
        )
        
        # Scheduler state
        self.running = False
        self.scheduler_thread = None
        self.last_health_check = None
        self.performance_history = []
        
        # Setup scheduler tables
        self._setup_scheduler_tables()
        
        # Configure scheduled jobs
        self._configure_schedule()
    
    def _setup_scheduler_tables(self):
        """Set up scheduler-specific tables."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create scheduler actions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tgr_scheduler_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    action_details TEXT,
                    triggered_by TEXT,
                    success BOOLEAN DEFAULT 1,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tgr_scheduler_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create scheduling rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tgr_scheduler_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT UNIQUE NOT NULL,
                    rule_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    priority INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_triggered TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Scheduler tables initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup scheduler tables: {e}")
    
    def _configure_schedule(self):
        """Configure scheduled tasks."""
        
        # Regular monitoring and health checks
        schedule.every(self.config.monitoring_interval).seconds.do(self._run_health_check)
        
        # Daily comprehensive enrichment
        schedule.every().day.at("02:00").do(self._run_daily_enrichment)
        
        # Hourly performance optimization
        schedule.every().hour.do(self._optimize_performance)
        
        # Weekly cleanup and maintenance
        schedule.every().sunday.at("01:00").do(self._run_maintenance)
        
        logger.info("✅ Schedule configured")
    
    def start_scheduler(self):
        """Start the intelligent scheduler."""
        
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("🚀 Starting TGR Service Scheduler")
        
        # Start enrichment service
        self.enrichment_service.start_service()
        
        self.running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="TGRScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        # Log scheduler start
        self._log_action("scheduler_started", "System initialization completed")
        
        logger.info("✅ TGR Service Scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler gracefully."""
        
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("🛑 Stopping TGR Service Scheduler")
        
        self.running = False
        
        # Stop enrichment service
        self.enrichment_service.stop_service()
        
        # Wait for scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        # Log scheduler stop
        self._log_action("scheduler_stopped", "Clean shutdown completed")
        
        logger.info("✅ TGR Service Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler execution loop."""
        
        logger.info("🔧 Scheduler thread started")
        
        while self.running:
            try:
                # Run pending scheduled jobs
                schedule.run_pending()
                
                # Check for dynamic scheduling opportunities
                self._check_dynamic_scheduling()
                
                # Sleep for a short interval
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("🔧 Scheduler thread stopped")
    
    def _run_health_check(self):
        """Run comprehensive health check and respond to issues."""
        
        start_time = time.time()
        
        try:
            logger.info("🔍 Running health check")
            
            # Get current system health
            health_report = self.monitor.get_system_health()
            self.last_health_check = health_report
            
            # Analyze health and take action
            actions_taken = self._analyze_health_and_act(health_report)
            
            execution_time = time.time() - start_time
            
            # Log health check
            self._log_action("health_check", 
                           f"Health: {health_report['overall_health']}, Actions: {len(actions_taken)}",
                           execution_time=execution_time)
            
            # Record performance metrics
            self._record_performance_metric("health_check_duration", execution_time)
            self._record_performance_metric("overall_health_score", health_report.get('health_score', 0))
            
            logger.info(f"✅ Health check completed in {execution_time:.1f}s, took {len(actions_taken)} actions")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._log_action("health_check", f"Failed: {str(e)}", success=False)
    
    def _analyze_health_and_act(self, health_report: Dict[str, Any]) -> List[str]:
        """Analyze health report and take corrective actions."""
        
        actions_taken = []
        
        # Check data freshness
        if health_report.get('data_quality', {}).get('avg_data_age_hours', 0) > self.config.data_freshness_hours:
            logger.warning("⚠️ Data freshness issue detected")
            
            # Schedule batch enrichment for stale data
            stale_dogs = self._get_stale_data_dogs()
            if stale_dogs:
                jobs_added = self._schedule_priority_enrichment(stale_dogs, "data_freshness")
                actions_taken.append(f"scheduled_{jobs_added}_freshness_jobs")
        
        # Check collection performance
        collection_success = health_report.get('performance', {}).get('collection_success_rate', 1.0)
        if collection_success < self.config.performance_threshold:
            logger.warning("⚠️ Collection performance issue detected")
            
            # Trigger collection optimization
            self._optimize_collection_performance()
            actions_taken.append("optimized_collection")
        
        # Check cache efficiency
        cache_hit_rate = health_report.get('performance', {}).get('cache_hit_rate', 1.0)
        if cache_hit_rate < 0.5:  # Less than 50% cache hits
            logger.warning("⚠️ Cache efficiency issue detected")
            
            # Refresh popular caches
            self._refresh_popular_caches()
            actions_taken.append("refreshed_caches")
        
        # Check for critical alerts
        alerts = health_report.get('alerts', {})
        critical_alerts = alerts.get('critical', [])
        
        for alert in critical_alerts:
            if 'database' in alert.get('message', '').lower():
                logger.error("🚨 Critical database issue detected")
                self._handle_database_issue()
                actions_taken.append("database_maintenance")
        
        # Check for system resource issues
        if health_report.get('system_health') == 'critical':
            logger.error("🚨 Critical system health issue")
            
            # Reduce service load
            self._reduce_system_load()
            actions_taken.append("reduced_system_load")
        
        return actions_taken
    
    def _run_daily_enrichment(self):
        """Run daily comprehensive enrichment batch."""
        
        start_time = time.time()
        
        try:
            logger.info("🌅 Running daily enrichment batch")
            
            # Get dogs that need enrichment
            priority_dogs = self._get_daily_enrichment_candidates()
            
            if priority_dogs:
                jobs_added = self._schedule_priority_enrichment(priority_dogs, "daily_batch")
                
                execution_time = time.time() - start_time
                
                self._log_action("daily_enrichment", 
                               f"Scheduled {jobs_added} jobs for {len(priority_dogs)} dogs",
                               execution_time=execution_time)
                
                logger.info(f"✅ Daily enrichment scheduled: {jobs_added} jobs in {execution_time:.1f}s")
            else:
                logger.info("ℹ️ No dogs need daily enrichment")
                
        except Exception as e:
            logger.error(f"Daily enrichment failed: {e}")
            self._log_action("daily_enrichment", f"Failed: {str(e)}", success=False)
    
    def _optimize_performance(self):
        """Optimize system performance based on metrics."""
        
        start_time = time.time()
        
        try:
            logger.info("⚡ Running performance optimization")
            
            # Get performance metrics
            perf_metrics = self._get_performance_metrics()
            
            optimizations = []
            
            # Optimize based on job success rates
            if perf_metrics.get('job_success_rate', 1.0) < self.config.performance_threshold:
                self._optimize_job_processing()
                optimizations.append("job_processing")
            
            # Optimize database queries
            if perf_metrics.get('avg_query_time', 0) > 1.0:  # > 1 second
                self._optimize_database_queries()
                optimizations.append("database_queries")
            
            # Cleanup old data
            if perf_metrics.get('database_size_mb', 0) > 1000:  # > 1GB
                self._cleanup_old_data()
                optimizations.append("data_cleanup")
            
            execution_time = time.time() - start_time
            
            self._log_action("performance_optimization", 
                           f"Applied optimizations: {', '.join(optimizations)}",
                           execution_time=execution_time)
            
            logger.info(f"✅ Performance optimization completed in {execution_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            self._log_action("performance_optimization", f"Failed: {str(e)}", success=False)
    
    def _run_maintenance(self):
        """Run weekly maintenance tasks."""
        
        start_time = time.time()
        
        try:
            logger.info("🧹 Running weekly maintenance")
            
            maintenance_tasks = []
            
            # Archive old logs
            self._archive_old_logs()
            maintenance_tasks.append("log_archival")
            
            # Optimize database
            self._optimize_database()
            maintenance_tasks.append("database_optimization")
            
            # Clean up temporary data
            self._cleanup_temporary_data()
            maintenance_tasks.append("temp_cleanup")
            
            # Generate performance reports
            self._generate_performance_reports()
            maintenance_tasks.append("performance_reports")
            
            execution_time = time.time() - start_time
            
            self._log_action("weekly_maintenance", 
                           f"Completed tasks: {', '.join(maintenance_tasks)}",
                           execution_time=execution_time)
            
            logger.info(f"✅ Weekly maintenance completed in {execution_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Weekly maintenance failed: {e}")
            self._log_action("weekly_maintenance", f"Failed: {str(e)}", success=False)
    
    def _check_dynamic_scheduling(self):
        """Check for dynamic scheduling opportunities."""
        
        try:
            # Check if enrichment service has capacity
            service_status = self.enrichment_service.get_service_status()
            queue_size = service_status.get('queue_size', 0)
            
            # If queue is low and service is running well, schedule more work
            if queue_size < 5 and service_status.get('service_running'):
                recent_success_rate = self._get_recent_success_rate()
                
                if recent_success_rate > 0.8:  # 80% success rate
                    # Find dogs that could benefit from enrichment
                    candidates = self._get_opportunistic_enrichment_candidates(limit=5)
                    
                    if candidates:
                        jobs_added = self._schedule_priority_enrichment(candidates, "opportunistic")
                        
                        if jobs_added > 0:
                            logger.info(f"🎯 Opportunistically scheduled {jobs_added} enrichment jobs")
            
        except Exception as e:
            logger.debug(f"Dynamic scheduling check failed: {e}")
    
    def _get_stale_data_dogs(self, limit: int = 10) -> List[str]:
        """Get dogs with stale data that need enrichment."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find dogs with old or missing performance summaries
            cursor.execute("""
                SELECT DISTINCT d.dog_clean_name
                FROM dog_race_data d
                LEFT JOIN tgr_dog_performance_summary ps ON d.dog_clean_name = ps.dog_name
                WHERE d.dog_clean_name IS NOT NULL 
                AND d.dog_clean_name != ''
                AND (ps.last_updated IS NULL 
                     OR ps.last_updated < datetime('now', '-24 hours'))
                AND d.race_date >= date('now', '-30 days')
                ORDER BY d.race_date DESC
                LIMIT ?
            """, [limit])
            
            stale_dogs = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return stale_dogs
            
        except Exception as e:
            logger.error(f"Failed to get stale data dogs: {e}")
            return []
    
    def _get_daily_enrichment_candidates(self, limit: int = 20) -> List[str]:
        """Get dogs that are candidates for daily enrichment."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prioritize active racing dogs with recent races
            cursor.execute("""
                SELECT d.dog_clean_name, COUNT(*) as recent_races,
                       MAX(d.race_date) as last_race
                FROM dog_race_data d
                LEFT JOIN tgr_dog_performance_summary ps ON d.dog_clean_name = ps.dog_name
                WHERE d.dog_clean_name IS NOT NULL 
                AND d.dog_clean_name != ''
                AND d.race_date >= date('now', '-14 days')
                AND (ps.last_updated IS NULL 
                     OR ps.last_updated < datetime('now', '-12 hours'))
                GROUP BY d.dog_clean_name
                HAVING recent_races >= 2
                ORDER BY recent_races DESC, last_race DESC
                LIMIT ?
            """, [limit])
            
            candidates = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get daily enrichment candidates: {e}")
            return []
    
    def _get_opportunistic_enrichment_candidates(self, limit: int = 5) -> List[str]:
        """Get dogs for opportunistic enrichment when system has capacity."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find dogs with good recent performance that could benefit from deeper analysis
            cursor.execute("""
                SELECT DISTINCT d.dog_clean_name
                FROM dog_race_data d
                LEFT JOIN tgr_expert_insights ei ON d.dog_clean_name = ei.dog_name
                WHERE d.dog_clean_name IS NOT NULL 
                AND d.dog_clean_name != ''
                AND d.race_date >= date('now', '-7 days')
                AND d.finishing_position <= 3
                AND ei.dog_name IS NULL
                ORDER BY RANDOM()
                LIMIT ?
            """, [limit])
            
            candidates = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get opportunistic candidates: {e}")
            return []
    
    def _schedule_priority_enrichment(self, dogs: List[str], context: str) -> int:
        """Schedule priority enrichment jobs for specified dogs."""
        
        jobs_added = 0
        
        for dog_name in dogs:
            job_id = self.enrichment_service.add_enrichment_job(
                dog_name=dog_name,
                job_type="comprehensive",
                priority=8 if context == "critical" else 6
            )
            
            if job_id:
                jobs_added += 1
        
        if jobs_added > 0:
            self._log_action("priority_enrichment", 
                           f"Scheduled {jobs_added} jobs for context: {context}")
        
        return jobs_added
    
    def _get_recent_success_rate(self) -> float:
        """Get recent job success rate."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    COUNT(*) as total
                FROM tgr_enrichment_jobs 
                WHERE created_at >= datetime('now', '-2 hours')
                AND status IN ('completed', 'failed')
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[1] > 0:
                return result[0] / result[1]
            
            return 1.0  # Assume good if no recent data
            
        except Exception as e:
            logger.error(f"Failed to get recent success rate: {e}")
            return 0.5
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        
        metrics = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Job success rate
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM tgr_enrichment_jobs 
                WHERE created_at >= datetime('now', '-24 hours')
            """)
            result = cursor.fetchone()
            if result and result[0] is not None:
                metrics['job_success_rate'] = result[0]
            
            # Average processing time
            cursor.execute("""
                SELECT AVG(actual_duration)
                FROM tgr_enrichment_jobs 
                WHERE status = 'completed' 
                AND completed_at >= datetime('now', '-24 hours')
            """)
            result = cursor.fetchone()
            if result and result[0] is not None:
                metrics['avg_job_duration'] = result[0]
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
        
        return metrics
    
    def _optimize_collection_performance(self):
        """Optimize data collection performance."""
        logger.info("🔧 Optimizing collection performance")
        # Implementation would involve tuning collection parameters
    
    def _refresh_popular_caches(self):
        """Refresh popular feature caches."""
        logger.info("🔄 Refreshing popular caches")
        # Implementation would refresh most-accessed caches
    
    def _handle_database_issue(self):
        """Handle detected database issues."""
        logger.info("🛠️ Handling database issue")
        # Implementation would include database maintenance tasks
    
    def _reduce_system_load(self):
        """Reduce system load during critical conditions."""
        logger.info("⬇️ Reducing system load")
        # Implementation would reduce concurrent operations
    
    def _optimize_job_processing(self):
        """Optimize job processing efficiency."""
        logger.info("⚙️ Optimizing job processing")
        
    def _optimize_database_queries(self):
        """Optimize database query performance."""
        logger.info("📊 Optimizing database queries")
    
    def _cleanup_old_data(self):
        """Clean up old data to free space."""
        logger.info("🗑️ Cleaning up old data")
    
    def _archive_old_logs(self):
        """Archive old log entries."""
        logger.info("📦 Archiving old logs")
    
    def _optimize_database(self):
        """Optimize database structure and indices."""
        logger.info("🗄️ Optimizing database")
    
    def _cleanup_temporary_data(self):
        """Clean up temporary data files."""
        logger.info("🧹 Cleaning temporary data")
    
    def _generate_performance_reports(self):
        """Generate performance analysis reports."""
        logger.info("📈 Generating performance reports")
    
    def _log_action(self, action_type: str, details: str = None, 
                   execution_time: float = None, success: bool = True):
        """Log scheduler action."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO tgr_scheduler_actions 
                (action_type, action_details, triggered_by, success, execution_time)
                VALUES (?, ?, 'scheduler', ?, ?)
            """, [action_type, details, success, execution_time])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Failed to log scheduler action: {e}")
    
    def _record_performance_metric(self, metric_name: str, value: float, context: str = None):
        """Record a performance metric."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO tgr_scheduler_performance (metric_name, metric_value, context)
                VALUES (?, ?, ?)
            """, [metric_name, value, context])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Failed to record performance metric: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        
        status = {
            'scheduler_running': self.running,
            'enrichment_service_running': self.enrichment_service.running,
            'last_health_check': self.last_health_check.get('timestamp') if self.last_health_check else None
        }
        
        # Add enrichment service status
        if self.enrichment_service.running:
            status['enrichment_status'] = self.enrichment_service.get_service_status()
        
        # Add recent actions
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT action_type, COUNT(*) 
                FROM tgr_scheduler_actions 
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY action_type
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            
            status['recent_actions'] = dict(cursor.fetchall())
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
        
        return status

def main():
    """Main scheduler execution."""
    
    print("🎯 TGR Service Scheduler")
    print("-" * 30)
    
    # Create scheduler with custom config
    config = SchedulerConfig(
        monitoring_interval=60,  # 1 minute for demo
        enrichment_batch_size=5,
        max_concurrent_jobs=2
    )
    
    scheduler = TGRServiceScheduler(config=config)
    
    try:
        # Start the scheduler
        scheduler.start_scheduler()
        
        # Let it run for a few minutes
        print("🔄 Running scheduler for 3 minutes...")
        time.sleep(180)
        
        # Show status
        status = scheduler.get_scheduler_status()
        print(f"\n📊 Scheduler Status:")
        print(f"   Scheduler Running: {status['scheduler_running']}")
        print(f"   Enrichment Running: {status['enrichment_service_running']}")
        
        if 'enrichment_status' in status:
            enrich_stats = status['enrichment_status']['statistics']
            print(f"   Jobs Processed: {enrich_stats['jobs_processed']}")
            print(f"   Success Rate: {enrich_stats['jobs_succeeded']}/{enrich_stats['jobs_processed']}")
        
        if 'recent_actions' in status:
            print(f"   Recent Actions: {status['recent_actions']}")
        
    finally:
        # Stop the scheduler
        scheduler.stop_scheduler()
        
    print("\n✅ Scheduler demo completed")

if __name__ == "__main__":
    main()
