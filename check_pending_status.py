#!/usr/bin/env python3
"""
Check Pending Race Status
========================

This utility script provides statistics about races with pending winner status
in the database.

Usage:
    python check_pending_status.py [--detailed]

Options:
    --detailed      Show detailed information including recent samples
    --help         Show this help message

Author: AI Assistant
Date: August 23, 2025
"""

import argparse
import sys
from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor


def main():
    """Main function to check pending race statistics"""
    parser = argparse.ArgumentParser(
        description='Check statistics about races with pending winner status',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--detailed', 
        action='store_true',
        help='Show detailed information including recent samples'
    )
    
    args = parser.parse_args()
    
    print("📊 GREYHOUND RACING PENDING STATUS CHECK")
    print("=" * 45)
    
    # Initialize processor (minimal mode - no web driver needed for stats)
    try:
        processor = EnhancedComprehensiveProcessor(processing_mode='minimal')
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return 1
    
    try:
        # Get pending race statistics
        stats = processor.get_pending_race_statistics()
        
        if 'error' in stats:
            print(f"❌ Error getting statistics: {stats['error']}")
            return 1
        
        # Display basic statistics
        total_pending = stats.get('total_pending', 0)
        total_complete = stats.get('total_complete', 0)
        completion_rate = stats.get('completion_rate', 0)
        
        print(f"📈 OVERALL STATISTICS")
        print(f"   🔄 Pending races: {total_pending}")
        print(f"   ✅ Complete races: {total_complete}")
        print(f"   📊 Completion rate: {completion_rate:.1%}")
        print()
        
        if total_pending == 0:
            print("🎉 All races have winners! No backfill needed.")
            return 0
        
        # Show breakdown by venue
        by_venue = stats.get('by_venue', {})
        if by_venue:
            print("🏆 PENDING BY VENUE:")
            for venue, count in sorted(by_venue.items(), key=lambda x: x[1], reverse=True):
                print(f"   {venue}: {count} races")
            print()
        
        # Show breakdown by scraping attempts
        by_attempts = stats.get('by_attempts', {})
        if by_attempts:
            print("🔁 PENDING BY ATTEMPTS:")
            for attempts, count in sorted(by_attempts.items()):
                if attempts == 0:
                    print(f"   Not attempted: {count} races")
                else:
                    print(f"   {attempts} attempt{'s' if attempts > 1 else ''}: {count} races")
            print()
        
        # Show detailed information if requested
        if args.detailed:
            recent_sample = stats.get('recent_sample', [])
            if recent_sample:
                print("📋 RECENT PENDING RACES (sample):")
                for race in recent_sample:
                    race_id = race.get('race_id', 'Unknown')
                    venue = race.get('venue', 'Unknown')
                    race_number = race.get('race_number', 'Unknown')
                    race_date = race.get('race_date', 'Unknown')
                    attempts = race.get('attempts', 0)
                    
                    attempts_str = f"{attempts} attempts" if attempts > 0 else "not attempted"
                    print(f"   📅 {race_date}: {venue} Race {race_number} - {attempts_str}")
                print()
        
        # Provide recommendations
        print("💡 RECOMMENDATIONS:")
        if total_pending > 0:
            print(f"   🔄 Run backfill process to attempt scraping winners for {total_pending} pending races")
            print(f"   ⚙️  Command: python run_backfill.py --max-races {min(total_pending, 50)}")
            
            # Check if there are races with max attempts
            max_attempts = max(by_attempts.keys()) if by_attempts else 0
            if max_attempts >= 3:
                max_attempt_count = by_attempts.get(max_attempts, 0)
                print(f"   ⚠️  {max_attempt_count} races have reached maximum attempts ({max_attempts})")
                print(f"   🔧 Consider increasing max-retries or manual intervention for these races")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Status check interrupted by user")
        return 130
        
    except Exception as e:
        print(f"❌ Unexpected error during status check: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        try:
            processor.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
