#!/usr/bin/env python3
"""
TGR Enrichment System Demo
=========================

Comprehensive demo of the complete TGR data enrichment system including:
- Monitoring Dashboard
- Enrichment Service  
- Intelligent Scheduler
- Integration showcase
"""

import time
import json
import threading
from datetime import datetime

# Import our TGR components
from tgr_monitoring_dashboard import TGRMonitoringDashboard
from tgr_enrichment_service import TGREnrichmentService

def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print('='*60)

def print_subsection(title: str):
    """Print a formatted subsection."""
    print(f"\n📊 {title}")
    print('-'*40)

def demo_monitoring_dashboard():
    """Demo the monitoring dashboard functionality."""
    print_section_header("TGR Monitoring Dashboard Demo")
    
    monitor = TGRMonitoringDashboard()
    
    print("🔍 Initializing monitoring dashboard...")
    
    # Generate comprehensive report
    print_subsection("System Health Check")
    report = monitor.generate_comprehensive_report()
    
    health = report['system_health']
    print(f"System Status: {health.get('status', 'unknown').upper()}")
    
    if 'uptime_metrics' in health:
        metrics = health['uptime_metrics']
        print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"Total Attempts (7d): {metrics.get('total_attempts_7d', 0)}")
    
    # Show data quality
    data_quality = report['data_quality']
    print(f"Data Quality Score: {data_quality.get('overall_score', 0):.1f}/100")
    
    if 'completeness' in data_quality:
        comp = data_quality['completeness']
        print(f"Complete Profiles: {comp.get('complete_profiles', 0)}/{comp.get('total_dogs', 0)}")
    
    # Show performance metrics
    print_subsection("Performance Metrics")
    performance = report['performance_metrics']
    
    if 'cache_performance' in performance:
        cache = performance['cache_performance']
        print(f"Cache Hit Rate: {cache.get('hit_rate', 0):.1f}%")
        print(f"Cache Entries: {cache.get('total_entries', 0)}")
    
    if 'collection_efficiency' in performance:
        collection = performance['collection_efficiency']
        print(f"Collections per Day: {collection.get('collections_per_day', 0):.1f}")
        print(f"Avg Collection Time: {collection.get('avg_collection_time_sec', 0):.1f}s")
    
    # Generate and display dashboard
    print_subsection("Live Dashboard")
    monitor.print_dashboard()
    
    # Export detailed report
    report_file = f"tgr_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    exported_file = monitor.export_report(report_file)
    
    if exported_file:
        print(f"\n✅ Detailed health report exported to: {exported_file}")
    
    return monitor

def demo_enrichment_service():
    """Demo the enrichment service functionality."""
    print_section_header("TGR Enrichment Service Demo")
    
    service = TGREnrichmentService(max_workers=1, batch_size=3)
    
    print("🚀 Starting enrichment service...")
    service.start_service()
    
    # Add some test jobs
    print_subsection("Adding Test Enrichment Jobs")
    
    test_dogs = [
        ("LIGHTNING BOLT", "comprehensive"),
        ("SPEEDY GONZALES", "performance_analysis"), 
        ("TRACK STAR", "expert_insights"),
        ("CHAMPION RUNNER", "comprehensive")
    ]
    
    job_ids = []
    for dog_name, job_type in test_dogs:
        job_id = service.add_enrichment_job(dog_name, job_type, priority=7)
        if job_id:
            job_ids.append(job_id)
            print(f"  ✅ Added {job_type} job for {dog_name}")
    
    # Let jobs process
    print(f"\n🔄 Processing {len(job_ids)} enrichment jobs...")
    time.sleep(10)  # Wait for some jobs to complete
    
    # Show service status
    print_subsection("Service Status")
    status = service.get_service_status()
    
    print(f"Service Running: {status['service_running']}")
    print(f"Active Workers: {status['active_workers']}")
    print(f"Queue Size: {status['queue_size']}")
    print(f"Jobs Processed: {status['statistics']['jobs_processed']}")
    print(f"Success Rate: {status['statistics']['jobs_succeeded']}/{status['statistics']['jobs_processed']}")
    
    if status['statistics']['total_processing_time'] > 0:
        avg_time = status['statistics']['total_processing_time'] / max(status['statistics']['jobs_processed'], 1)
        print(f"Average Job Time: {avg_time:.1f}s")
    
    # Schedule batch enrichment
    print_subsection("Batch Enrichment")
    batch_jobs = service.schedule_batch_enrichment()
    print(f"Scheduled {batch_jobs} batch enrichment jobs")
    
    # Wait a bit more for processing
    time.sleep(10)
    
    # Final status
    final_status = service.get_service_status()
    print(f"\n📈 Final Statistics:")
    print(f"  Total Jobs Processed: {final_status['statistics']['jobs_processed']}")
    print(f"  Success Rate: {final_status['statistics']['jobs_succeeded']}/{final_status['statistics']['jobs_processed']}")
    print(f"  Total Processing Time: {final_status['statistics']['total_processing_time']:.1f}s")
    
    print("\n🛑 Stopping enrichment service...")
    service.stop_service()
    
    return service

def demo_scheduler_integration():
    """Demo the intelligent scheduler integration."""
    print_section_header("Intelligent Scheduler Demo")
    
    print("⚠️ Note: Full scheduler demo requires extended runtime.")
    print("Showing scheduler initialization and configuration...")
    
    # This would normally run the full scheduler, but we'll just show setup
    try:
        from tgr_service_scheduler import TGRServiceScheduler, SchedulerConfig
        
        # Create scheduler config
        config = SchedulerConfig(
            monitoring_interval=60,
            enrichment_batch_size=5,
            max_concurrent_jobs=2,
            performance_threshold=0.8
        )
        
        # Initialize scheduler (but don't run full demo)
        scheduler = TGRServiceScheduler(config=config)
        
        print("✅ Scheduler initialized successfully")
        print(f"  Monitoring Interval: {config.monitoring_interval}s")
        print(f"  Batch Size: {config.enrichment_batch_size}")
        print(f"  Max Concurrent Jobs: {config.max_concurrent_jobs}")
        print(f"  Performance Threshold: {config.performance_threshold:.0%}")
        
        # Show what the scheduler would do
        print("\n📋 Scheduler Capabilities:")
        print("  • Automated health monitoring")
        print("  • Intelligent job scheduling")
        print("  • Performance optimization")
        print("  • Predictive workload management")
        print("  • Error recovery and retry logic")
        print("  • Resource optimization")
        print("  • Data freshness maintenance")
        
    except ImportError as e:
        print(f"⚠️ Scheduler demo limited due to missing dependency: {e}")

def demo_data_quality_improvements():
    """Demo improvements in data quality through enrichment."""
    print_section_header("Data Quality Improvements Demo")
    
    import sqlite3
    
    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()
        
        # Check enhanced TGR data
        print_subsection("Enhanced TGR Data Analysis")
        
        # Count performance summaries
        cursor.execute("SELECT COUNT(*) FROM tgr_dog_performance_summary")
        perf_count = cursor.fetchone()[0]
        print(f"Dogs with Performance Summaries: {perf_count}")
        
        # Count expert insights  
        cursor.execute("SELECT COUNT(*) FROM tgr_expert_insights")
        insights_count = cursor.fetchone()[0]
        print(f"Expert Insights Generated: {insights_count}")
        
        # Count feature cache entries
        cursor.execute("SELECT COUNT(*) FROM tgr_enhanced_feature_cache")
        cache_count = cursor.fetchone()[0]
        print(f"Feature Cache Entries: {cache_count}")
        
        # Show sample enhanced data
        cursor.execute("""
            SELECT dog_name, consistency_score, form_trend, win_percentage 
            FROM tgr_dog_performance_summary 
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print_subsection("Sample Enhanced Performance Data")
            print(f"{'Dog Name':<20} {'Consistency':<12} {'Form Trend':<12} {'Win %':<8}")
            print("-" * 55)
            
            for dog_name, consistency, trend, win_pct in sample_data:
                consistency = f"{consistency:.1f}" if consistency else "N/A"
                win_pct = f"{win_pct:.1f}%" if win_pct else "N/A"
                trend = trend or "N/A"
                print(f"{dog_name:<20} {consistency:<12} {trend:<12} {win_pct:<8}")
        
        # Show enrichment job statistics
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM tgr_enrichment_jobs 
            GROUP BY status
        """)
        job_stats = cursor.fetchall()
        
        if job_stats:
            print_subsection("Enrichment Job Statistics")
            for status, count in job_stats:
                print(f"  {status.capitalize()}: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"⚠️ Could not analyze data quality: {e}")

def demo_system_integration():
    """Demo how all components work together."""
    print_section_header("System Integration Overview")
    
    print("🔗 The TGR Enrichment System Components:")
    print()
    
    print("1. 📊 MONITORING DASHBOARD")
    print("   • Real-time health monitoring")
    print("   • Performance metrics tracking")
    print("   • Alert generation")
    print("   • Quality assessment")
    print()
    
    print("2. ⚙️ ENRICHMENT SERVICE")
    print("   • Multi-threaded job processing")
    print("   • Intelligent retry logic")
    print("   • Performance tracking")
    print("   • Priority-based scheduling")
    print()
    
    print("3. 🎯 INTELLIGENT SCHEDULER")
    print("   • Health-based decision making")
    print("   • Predictive workload management")
    print("   • Automatic optimization")
    print("   • Resource management")
    print()
    
    print("4. 📈 DATA ENRICHMENT PIPELINE")
    print("   • Performance analysis")
    print("   • Expert insights generation")
    print("   • Feature cache management")
    print("   • Venue/distance analytics")
    print()
    
    print("🔄 Integration Flow:")
    print("   Monitor → Assess → Schedule → Enrich → Validate → Optimize")

def main():
    """Main demo execution."""
    print("🚀 TGR Data Enrichment System")
    print("=" * 60)
    print("Comprehensive demonstration of automated TGR data enrichment")
    print("and monitoring capabilities for greyhound racing predictions.")
    
    try:
        # Demo each component
        monitor = demo_monitoring_dashboard()
        time.sleep(2)
        
        enrichment_service = demo_enrichment_service()
        time.sleep(2)
        
        demo_scheduler_integration()
        time.sleep(2)
        
        demo_data_quality_improvements()
        time.sleep(2)
        
        demo_system_integration()
        
        print_section_header("Demo Complete")
        print("✅ All TGR enrichment system components demonstrated successfully!")
        print()
        print("🎯 Key Achievements:")
        print("  • Automated data quality monitoring")
        print("  • Intelligent enrichment job processing")
        print("  • Performance-based optimization")
        print("  • Enhanced predictive features")
        print("  • Scalable system architecture")
        print()
        print("📝 Next Steps:")
        print("  • Deploy scheduler for continuous operation")
        print("  • Configure monitoring thresholds") 
        print("  • Set up performance alerting")
        print("  • Integrate with ML pipeline")
        print("  • Schedule regular maintenance")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n🏁 Demo completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
