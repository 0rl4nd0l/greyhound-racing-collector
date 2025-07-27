#!/usr/bin/env python3
"""
Continuous Odds Monitoring Script
=================================

This script runs continuous monitoring of live odds from Sportsbet,
updating the database every 30 seconds and identifying value betting opportunities.

Features:
- Continuous odds monitoring
- Real-time database updates
- Value bet detection
- Performance metrics
- Error handling and recovery
- Graceful shutdown handling
"""

import signal
import sys
import time
import json
from datetime import datetime, timedelta
from sportsbet_odds_integrator import SportsbetOddsIntegrator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('odds_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OddsMonitor:
    """Continuous odds monitoring system"""
    
    def __init__(self, update_interval=30):
        self.integrator = SportsbetOddsIntegrator()
        self.update_interval = update_interval
        self.running = False
        self.stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'total_races_processed': 0,
            'total_odds_collected': 0,
            'value_bets_found': 0,
            'start_time': None,
            'last_update': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def update_odds_cycle(self):
        """Perform one complete odds update cycle"""
        cycle_start = datetime.now()
        logger.info("🔄 Starting odds update cycle...")
        
        try:
            # Get today's races with live odds
            races = self.integrator.get_today_races()
            self.stats['total_races_processed'] += len(races)
            
            # Save odds to database
            odds_count = 0
            for race in races:
                self.integrator.save_odds_to_database(race)
                odds_data = race.get('odds_data', [])
                odds_count += len(odds_data)
            
            self.stats['total_odds_collected'] += odds_count
            
            # Identify value bets
            value_bets = self.integrator.identify_value_bets()
            self.stats['value_bets_found'] += len(value_bets)
            
            # Log cycle results
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"✅ Cycle complete: {len(races)} races, {odds_count} odds, {len(value_bets)} value bets ({cycle_duration:.1f}s)")
            
            self.stats['successful_updates'] += 1
            self.stats['last_update'] = datetime.now()
            
            # Log any significant value bets
            if value_bets:
                logger.info(f"💰 Value betting opportunities found:")
                for bet in value_bets[:3]:  # Log top 3
                    logger.info(f"  {bet['dog_clean_name']}: {bet['bet_recommendation']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Odds update cycle failed: {e}")
            self.stats['failed_updates'] += 1
            return False
    
    def print_statistics(self):
        """Print current monitoring statistics"""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            runtime_hours = runtime.total_seconds() / 3600
            
            success_rate = (self.stats['successful_updates'] / max(1, self.stats['total_updates'])) * 100
            
            logger.info("📊 Monitoring Statistics:")
            logger.info(f"  Runtime: {runtime}")
            logger.info(f"  Total Updates: {self.stats['total_updates']}")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            logger.info(f"  Races Processed: {self.stats['total_races_processed']}")
            logger.info(f"  Odds Collected: {self.stats['total_odds_collected']}")
            logger.info(f"  Value Bets Found: {self.stats['value_bets_found']}")
            logger.info(f"  Last Update: {self.stats['last_update']}")
            
            if runtime_hours > 0:
                logger.info(f"  Avg Races/Hour: {self.stats['total_races_processed']/runtime_hours:.1f}")
                logger.info(f"  Avg Odds/Hour: {self.stats['total_odds_collected']/runtime_hours:.1f}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("🚀 Starting continuous odds monitoring...")
        logger.info(f"📅 Update interval: {self.update_interval} seconds")
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Initial update
        self.stats['total_updates'] += 1
        self.update_odds_cycle()
        
        try:
            while self.running:
                # Wait for next update
                time.sleep(self.update_interval)
                
                if not self.running:
                    break
                
                # Check if we should still be monitoring (racing hours)
                current_hour = datetime.now().hour
                if current_hour < 6 or current_hour > 23:
                    logger.info("⏰ Outside racing hours, reducing update frequency...")
                    time.sleep(300)  # Wait 5 minutes during off-hours
                    continue
                
                # Perform update cycle
                self.stats['total_updates'] += 1
                self.update_odds_cycle()
                
                # Print stats every 10 updates
                if self.stats['total_updates'] % 10 == 0:
                    self.print_statistics()
                    
        except KeyboardInterrupt:
            logger.info("⚠️  Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down odds monitor...")
        self.running = False
        
        # Close integrator resources
        self.integrator.close_driver()
        
        # Print final statistics
        self.print_statistics()
        
        # Save final stats to file
        stats_file = f"monitor_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(stats_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                stats_copy = self.stats.copy()
                for key, value in stats_copy.items():
                    if isinstance(value, datetime):
                        stats_copy[key] = value.isoformat()
                json.dump(stats_copy, f, indent=2)
            logger.info(f"📊 Statistics saved to {stats_file}")
        except Exception as e:
            logger.error(f"❌ Error saving statistics: {e}")
        
        logger.info("✅ Shutdown complete")

def run_single_update():
    """Run a single odds update (for automation)"""
    logger.info("🎯 Running single odds update...")
    
    monitor = OddsMonitor(update_interval=30)
    monitor.stats['start_time'] = datetime.now()
    monitor.stats['total_updates'] = 1
    
    success = monitor.update_odds_cycle()
    
    # Print summary
    if success:
        logger.info("✅ Single odds update completed successfully")
    else:
        logger.error("❌ Single odds update failed")
        sys.exit(1)
    
    # Clean up
    monitor.integrator.close_driver()
    return success

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Greyhound Racing Odds Monitor')
    parser.add_argument('--single-run', action='store_true', 
                       help='Run a single odds update and exit (for automation)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    if args.single_run:
        return run_single_update()
    
    # Continuous monitoring mode
    logger.info("🏁 Odds Monitor Starting...")
    logger.info(f"🔧 Update interval: {args.interval} seconds")
    
    # Create and start monitor
    monitor = OddsMonitor(update_interval=args.interval)
    
    try:
        monitor.start_monitoring()
    except Exception as e:
        logger.error(f"❌ Monitor failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
