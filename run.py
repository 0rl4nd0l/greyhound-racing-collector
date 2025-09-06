#!/usr/bin/env python3
"""
Main Entry Point Script
========================

This script provides the main interface for running data collection and analysis tasks.
It's called by the Flask app for various background operations.

Usage:
    python run.py collect    # Run data collection
    python run.py analyze    # Run data analysis
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import profiling configuration
from profiling_config import is_profiling, set_profiling_enabled


def run_collection():
    """Run data collection process"""
    print("🔍 Starting data collection...")

    # Ensure comprehensive scraping of all available data using expert form method
    try:
        print("📊 Running expert form CSV scraper for enhanced data collection...")
        result = subprocess.run(
            [
                sys.executable,
                "expert_form_csv_scraper.py",
                "--days-ahead",
                "2",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=180,  # Extended timeout for expert form scraping
        )
        if result.returncode == 0:
            print("✅ Expert form CSV scraping completed successfully")
            # Print some stats from the output if available
            if result.stdout:
                print(f"📊 Scraper output (last 200 chars): {result.stdout[-200:]}")
        else:
            print(
                f"⚠️ Expert form CSV scraping issues: {result.stderr[:200] if result.stderr else 'Unknown error'}"
            )
    except subprocess.TimeoutExpired:
        print("⏰ Expert form CSV scraping timed out, please retry if needed.")
    except Exception as e:
        print(f"❌ Expert form CSV scraping failed: {e}")

    # Check for upcoming race browser
    if os.path.exists("upcoming_race_browser.py"):
        try:
            print("🏁 Collecting upcoming races...")
            from upcoming_race_browser import UpcomingRaceBrowser

            browser = UpcomingRaceBrowser()
            races = browser.get_upcoming_races(days_ahead=1)
            print(f"✅ Found {len(races)} upcoming races")

            # Move downloaded races to unprocessed for automatic processing
            upcoming_dir = Path("./upcoming_races")
            unprocessed_dir = Path("./unprocessed")

            if upcoming_dir.exists():
                upcoming_files = list(upcoming_dir.glob("*.csv"))
                for file in upcoming_files:
                    destination = unprocessed_dir / file.name
                    if destination.exists():
                        print(f"   ⏭️ Skipping {file.name} - already in unprocessed")
                    else:
                        file.rename(destination)
                        print(f"   📂 Moved {file.name} to unprocessed for processing")
        except Exception as e:
            print(f"⚠️ Upcoming race collection had issues: {e}")

    print("🏁 Data collection completed")


def run_analysis(strict_scan: bool = False):
    """Run data analysis process with mtime optimization"""
    print("📈 Starting data analysis...")

    # Check for unprocessed files using mtime heuristic
    unprocessed_dir = "./unprocessed"
    if not os.path.exists(unprocessed_dir):
        print("⚠️ No unprocessed directory found")
        return

    # Use mtime heuristic for efficient file scanning
    try:
        from utils.mtime_heuristic import create_mtime_heuristic

        heuristic = create_mtime_heuristic()

        # Get files that need processing using mtime optimization
        files_to_process = list(
            heuristic.scan_directory_optimized(
                unprocessed_dir, strict_scan=strict_scan, file_extensions=[".csv"]
            )
        )

        if not files_to_process:
            if strict_scan:
                print("ℹ️ No CSV files found in unprocessed directory")
            else:
                print(
                    "ℹ️ No new files to process (use --strict-scan to force full re-scan)"
                )
            return

        print(f"📊 Found {len(files_to_process)} files to process")
        if not strict_scan:
            stats = heuristic.get_scan_statistics()
            if stats.get("heuristic_enabled"):
                print(
                    f"🚀 Using mtime optimization (last processed: {stats.get('last_processed_datetime', 'N/A')})"
                )

        # Convert FileEntry objects to file paths for compatibility
        unprocessed_files = [entry.name for entry in files_to_process]

    except ImportError as e:
        print(f"⚠️ Mtime heuristic not available, falling back to full scan: {e}")
        # Fallback to traditional directory listing
        unprocessed_files = [
            f for f in os.listdir(unprocessed_dir) if f.endswith(".csv")
        ]
        if not unprocessed_files:
            print("ℹ️ No unprocessed files found")
            return
        print(f"📊 Found {len(unprocessed_files)} files to process (full scan)")

    # Try to use enhanced comprehensive processor
    if os.path.exists("enhanced_comprehensive_processor.py"):
        try:
            print("🔧 Running enhanced comprehensive processor...")
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "enhanced_comprehensive_processor",
                "./enhanced_comprehensive_processor.py",
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                processor = module.EnhancedComprehensiveProcessor()
                results = processor.process_all_unprocessed()

                if results.get("status") == "success":
                    print(
                        f"✅ Processing completed! Processed {results.get('processed_count', 0)} files"
                    )
                else:
                    print(
                        f"❌ Processing failed: {results.get('message', 'Unknown error')}"
                    )
            else:
                print("❌ Could not load enhanced processor")
        except Exception as e:
            print(f"❌ Enhanced processing failed: {e}")
            # Fallback to basic file moving
            print("🔄 Using basic file processing...")
            basic_file_processing()
    else:
        print("🔄 Using basic file processing...")
        basic_file_processing()

    print("🏁 Data analysis completed")


def basic_file_processing():
    """Basic file processing fallback"""
    import shutil

    unprocessed_dir = "./unprocessed"
    processed_dir = "./processed"

    os.makedirs(processed_dir, exist_ok=True)

    unprocessed_files = [f for f in os.listdir(unprocessed_dir) if f.endswith(".csv")]
    processed_count = 0
    processed_file_paths = []

    for filename in unprocessed_files:
        try:
            source_path = os.path.join(unprocessed_dir, filename)
            dest_path = os.path.join(processed_dir, filename)

            if os.path.exists(dest_path):
                print(f"⚠️ {filename} already processed, skipping")
                continue

            shutil.copy2(source_path, dest_path)
            os.remove(source_path)
            processed_count += 1
            processed_file_paths.append(source_path)  # Track for mtime update
            print(f"✅ {filename} processed")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Update mtime heuristic with processed files
    if processed_file_paths:
        try:
            from utils.mtime_heuristic import create_mtime_heuristic

            heuristic = create_mtime_heuristic()
            heuristic.update_processed_mtime_from_files(processed_file_paths)
            print(
                f"🚀 Updated mtime heuristic with {len(processed_file_paths)} processed files"
            )
        except Exception as e:
            print(f"⚠️ Mtime heuristic update failed (non-critical): {e}")

    print(f"✅ Basic processing completed! Processed {processed_count} files")


def run_prediction(race_file_path=None):
    """Run prediction process"""
    print("🎯 Starting prediction process...")

    # Use comprehensive prediction pipeline
    try:
        from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline

        pipeline = ComprehensivePredictionPipeline()

        if race_file_path:
            # Predict specific file
            results = pipeline.predict_race_file(race_file_path)

            if results["success"]:
                print("✅ Prediction completed successfully!")
                print(
                    f"🏆 Top pick: {results['predictions'][0]['dog_name'] if results['predictions'] else 'None'}"
                )
                return True
            else:
                print(f"❌ Prediction failed: {results['error']}")
                return False
        else:
            # Use batch prediction for all upcoming races
            results = pipeline.predict_all_upcoming_races(
                upcoming_dir="./upcoming_races"
            )

            if results["success"]:
                print(f"✅ Batch prediction completed successfully!")
                print(
                    f"📊 Processed {results['total_races']} races with {results['successful_predictions']} successful predictions"
                )
                return results["successful_predictions"] > 0
            else:
                print(
                    f"❌ Batch prediction failed: {results.get('message', 'Unknown error')}"
                )
                return False

    except ImportError as e:
        print(f"⚠️ Comprehensive prediction pipeline not available: {e}")

        # Fallback to existing predictor
        if os.path.exists("upcoming_race_predictor.py"):
            print("🔄 Using fallback predictor...")
            result = subprocess.run(
                [sys.executable, "upcoming_race_predictor.py"]
                + ([race_file_path] if race_file_path else []),
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0
        else:
            print("❌ No prediction system available")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def create_parser():
    """Create argument parser for the main CLI"""
    parser = argparse.ArgumentParser(
        description="Greyhound Racing Data Collection and Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py collect --enable-profiling
  python run.py analyze                    # Use mtime optimization
  python run.py analyze --strict-scan      # Disable mtime heuristic for full re-scan
  python run.py predict race.csv --enable-profiling
        """,
    )

    parser.add_argument(
        "command", choices=["collect", "analyze", "predict"], help="Command to execute"
    )

    parser.add_argument(
        "race_file_path", nargs="?", help="Path to race file (for predict command)"
    )

    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable profiling for performance analysis (default: disabled for zero overhead)",
    )

    parser.add_argument(
        "--strict-scan",
        action="store_true",
        help="Disable mtime heuristic for full file re-scan (default: use mtime optimization)",
    )

    return parser


def main():
    """Main entry point with profiling support"""
    parser = create_parser()
    args = parser.parse_args()

    # Configure profiling based on CLI flag
    if args.enable_profiling:
        set_profiling_enabled(True)
        print("🔍 Profiling enabled")
    else:
        set_profiling_enabled(False)

    # Show profiling status for debugging
    if is_profiling():
        print("📊 Running with profiling enabled")

    # Execute command
    if args.command == "collect":
        run_collection()
    elif args.command == "analyze":
        run_analysis(strict_scan=args.strict_scan)
    elif args.command == "predict":
        success = run_prediction(args.race_file_path)
        sys.exit(0 if success else 1)
    else:
        # This shouldn't happen due to argparse choices, but keeping for safety
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
