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

import os
import subprocess
import sys
from pathlib import Path


def run_collection():
    """Run data collection process"""
    print("🔍 Starting data collection...")

    # Ensure comprehensive scraping of all available data
    try:
        print("📊 Running form guide CSV scraper for full data...")
        result = subprocess.run(
            [sys.executable, "form_guide_csv_scraper.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )  # Extended timeout for full scrape
        if result.returncode == 0:
            print("✅ Form guide scraping completed successfully")
        else:
            print(f"⚠️ Form guide scraping issues: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("⏰ Form guide scraping timed out, please retry if needed.")
    except Exception as e:
        print(f"❌ Form guide scraping failed: {e}")

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


def run_analysis():
    """Run data analysis process"""
    print("📈 Starting data analysis...")

    # Check for unprocessed files
    unprocessed_dir = "./unprocessed"
    if not os.path.exists(unprocessed_dir):
        print("⚠️ No unprocessed directory found")
        return

    unprocessed_files = [f for f in os.listdir(unprocessed_dir) if f.endswith(".csv")]
    if not unprocessed_files:
        print("ℹ️ No unprocessed files found")
        return

    print(f"📊 Found {len(unprocessed_files)} files to process")

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
            print(f"✅ {filename} processed")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✅ Basic processing completed! Processed {processed_count} files")


def run_prediction(race_file_path=None):
    """Run prediction process"""
    print("🎯 Starting prediction process...")

    # Use comprehensive prediction pipeline
    try:
        from comprehensive_prediction_pipeline import \
            ComprehensivePredictionPipeline

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


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python run.py [collect|analyze|predict] [race_file_path]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "collect":
        run_collection()
    elif command == "analyze":
        run_analysis()
    elif command == "predict":
        race_file_path = sys.argv[2] if len(sys.argv) > 2 else None
        success = run_prediction(race_file_path)
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print("Usage: python run.py [collect|analyze|predict] [race_file_path]")
        sys.exit(1)


if __name__ == "__main__":
    main()
