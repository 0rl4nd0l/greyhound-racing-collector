#!/usr/bin/env python3
"""
Backfill Winners for Pending Races
==================================

This utility script runs the backfill process to attempt to scrape winners
for races that currently have 'pending' status in the database.

Usage:
    python run_backfill.py [--max-races N] [--max-retries N]

Options:
    --max-races N     Maximum number of pending races to process (default: 50)
    --max-retries N   Maximum retry attempts per race (default: 3)
    --help           Show this help message

Author: AI Assistant
Date: August 23, 2025
"""

import argparse
import sys
from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor


def main():
    """Main function to run the backfill process"""
    parser = argparse.ArgumentParser(
        description='Backfill winners for races with pending status',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--max-races', 
        type=int, 
        default=50,
        help='Maximum number of pending races to process (default: 50)'
    )
    
    parser.add_argument(
        '--max-retries', 
        type=int, 
        default=3,
        help='Maximum retry attempts per race (default: 3)'
    )
    
    parser.add_argument(
        '--processing-mode',
        choices=['full', 'fast', 'minimal'],
        default='full',
        help='Processing mode (default: full)'
    )
    
    args = parser.parse_args()
    
    print("🔄 GREYHOUND RACING WINNER BACKFILL UTILITY")
    print("=" * 50)
    print(f"📊 Max races to process: {args.max_races}")
    print(f"🔁 Max retries per race: {args.max_retries}")
    print(f"⚙️ Processing mode: {args.processing_mode}")
    print()
    
    # Initialize processor
    try:
        processor = EnhancedComprehensiveProcessor(processing_mode=args.processing_mode)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return 1
    
    try:
        # Run the backfill process
        results = processor.backfill_winners_for_pending_races(
            max_races=args.max_races,
            max_retries_per_race=args.max_retries
        )
        
        # Display results
        print(f"\n📈 BACKFILL COMPLETE")
        print("=" * 50)
        
        if results.get('status') == 'success':
            print(f"✅ Successfully backfilled: {results.get('backfilled_count', 0)} races")
            print(f"❌ Failed to backfill: {results.get('failed_count', 0)} races")
            print(f"⏭️ Skipped (max retries): {results.get('skipped_count', 0)} races")
            
            # Show individual results if there are any
            individual_results = results.get('results', [])
            if individual_results:
                print(f"\n📋 Individual Results:")
                for result in individual_results[:10]:  # Show first 10 results
                    race_id = result.get('race_id', 'Unknown')
                    status = result.get('status', 'unknown')
                    
                    if status == 'success':
                        winner = result.get('winner', 'Unknown')
                        attempts = result.get('attempts', 'Unknown')
                        print(f"   ✅ {race_id}: Winner = {winner} (attempt {attempts})")
                    elif status == 'failed':
                        reason = result.get('reason', 'Unknown reason')
                        attempts = result.get('attempts', 'Unknown')
                        print(f"   ❌ {race_id}: {reason} (attempt {attempts})")
                    elif status == 'skipped':
                        reason = result.get('reason', 'Unknown reason')
                        print(f"   ⏭️ {race_id}: {reason}")
                    elif status == 'error':
                        error = result.get('error', 'Unknown error')
                        attempts = result.get('attempts', 'Unknown')
                        print(f"   💥 {race_id}: {error} (attempt {attempts})")
                
                if len(individual_results) > 10:
                    print(f"   ... and {len(individual_results) - 10} more results")
            
            return_code = 0
            
        elif results.get('status') == 'skipped':
            reason = results.get('reason', 'Unknown reason')
            print(f"⏭️ Backfill was skipped: {reason}")
            return_code = 0
            
        else:
            error = results.get('error', 'Unknown error')
            print(f"❌ Backfill failed: {error}")
            return_code = 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 Backfill interrupted by user")
        return_code = 130
        
    except Exception as e:
        print(f"❌ Unexpected error during backfill: {e}")
        import traceback
        traceback.print_exc()
        return_code = 1
        
    finally:
        # Cleanup
        try:
            processor.cleanup()
        except:
            pass
    
    print(f"\n💡 Backfill process completed.")
    print("📋 Check the database for updated winner information.")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
