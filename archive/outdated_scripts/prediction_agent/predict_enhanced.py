#!/usr/bin/env python3
"""
Enhanced Greyhound Racing Predictor
==================================

Features:
- Automatically uses the most recent analysis/insights file
- Automatically copies CSV files to analysis_agent/unprocessed/ folder
- Provides comprehensive predictions with confidence weighting
- Multi-track support with venue-specific analysis

Usage: python3 predict_enhanced.py <race_file>

Author: AI Assistant
Date: July 11, 2025
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dynamic_prediction import print_dynamic_results, run_dynamic_prediction
from insights_loader import InsightsLoader


def copy_to_unprocessed(csv_file_path):
    """Copy uploaded CSV file to analysis_agent/unprocessed folder"""

    # Define paths
    unprocessed_dir = "../analysis_agent/unprocessed"

    # Create unprocessed directory if it doesn't exist
    os.makedirs(unprocessed_dir, exist_ok=True)

    # Get filename and create timestamped version
    filename = os.path.basename(csv_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create new filename with timestamp
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"

    # Copy file to unprocessed folder
    destination = os.path.join(unprocessed_dir, new_filename)

    try:
        shutil.copy2(csv_file_path, destination)
        print(f"✅ CSV copied to analysis_agent dataset: {new_filename}")
        return destination
    except Exception as e:
        print(f"❌ Error copying CSV to analysis_agent dataset: {e}")
        return None


def trigger_dataset_update():
    """Trigger automatic dataset update if possible"""

    # Check if analysis_agent directory exists
    analysis_dir = "../analysis_agent"
    if not os.path.exists(analysis_dir):
        print("ℹ️  Analysis agent directory not found - skipping dataset update")
        return False

    # Check for update scripts
    update_scripts = [
        os.path.join(analysis_dir, "update_dataset.py"),
        os.path.join(analysis_dir, "process_unprocessed.py"),
        os.path.join(analysis_dir, "auto_update.py"),
    ]

    for script in update_scripts:
        if os.path.exists(script):
            print(f"🔄 Found update script: {os.path.basename(script)}")
            print("💡 You can run it manually to update the dataset:")
            print(f"   cd ../analysis_agent && python {os.path.basename(script)}")
            return True

    print("ℹ️  No automatic update scripts found in analysis_agent")
    return False


def main():
    print("🏁 ENHANCED GREYHOUND RACING PREDICTOR")
    print("=" * 60)
    print("🎯 Features: Dynamic insights + Auto dataset copying")
    print()

    if len(sys.argv) < 2:
        print("📋 Usage: python3 predict_enhanced.py <race_file>")
        print("\n📁 Available files:")

        # Check form_guides directory
        if os.path.exists("form_guides"):
            print("   📂 form_guides/")
            for file in sorted(os.listdir("form_guides")):
                if file.endswith(".csv"):
                    print(f"     • {file}")

        # Check Downloads directory
        downloads_dir = os.path.expanduser("~/Downloads")
        if os.path.exists(downloads_dir):
            race_files = [
                f
                for f in os.listdir(downloads_dir)
                if f.endswith(".csv") and "Race" in f
            ]
            if race_files:
                print(f"\n   📥 ~/Downloads/ (race files):")
                for file in sorted(race_files)[:5]:  # Show last 5
                    print(f"     • {file}")

        print(f"\n💡 Examples:")
        print(f'   python3 predict_enhanced.py "Race 3 - HEA - 11 July 2025.csv"')
        print(
            f'   python3 predict_enhanced.py "~/Downloads/Race 3 - HEA - 11 July 2025.csv"'
        )
        print(
            f'   python3 predict_enhanced.py "form_guides/Race 2 - HEA - 11 July 2025.csv"'
        )
        return

    race_file = sys.argv[1]

    # Handle different file locations
    possible_paths = [
        race_file,  # Direct path
        os.path.join("form_guides", race_file),  # form_guides folder
        os.path.expanduser(f"~/Downloads/{race_file}"),  # Downloads folder
        os.path.expanduser(race_file),  # Expand ~ paths
    ]

    csv_file = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_file = path
            break

    if csv_file is None:
        print(f"❌ Error: '{race_file}' not found")
        print("🔍 Checked locations:")
        for path in possible_paths:
            print(f"   • {path}")
        return

    print(f"📊 Analyzing: {os.path.basename(csv_file)}")
    print(f"📂 Location: {csv_file}")
    print()

    # Step 1: Show available insights
    print("🔍 CHECKING AVAILABLE INSIGHTS")
    print("=" * 60)
    loader = InsightsLoader()
    loader.list_available_insights()
    print()

    # Step 2: Copy to unprocessed folder
    print("📋 COPYING TO ANALYSIS DATASET")
    print("=" * 60)
    copied_file = copy_to_unprocessed(csv_file)
    if copied_file:
        trigger_dataset_update()
    print()

    # Step 3: Run dynamic prediction
    print("🎯 RUNNING DYNAMIC PREDICTION")
    print("=" * 60)
    try:
        predictions, metadata = run_dynamic_prediction(csv_file)
        if predictions:
            print_dynamic_results(predictions, metadata)

            # Additional summary
            print(f"\n🎯 ENHANCED PREDICTION SUMMARY")
            print("=" * 60)
            print(
                f"✅ Used insights: {metadata['file_name']} ({metadata['data_type']})"
            )
            print(f"✅ Analysis type: {metadata['data_type'].upper()}")
            print(f"✅ Dataset copy: {'Success' if copied_file else 'Failed'}")
            print(f"✅ Predictions: {len(predictions)} greyhounds analyzed")

            # Show top 3 picks
            print(f"\n🏆 TOP 3 PICKS:")
            for i, pred in enumerate(predictions[:3], 1):
                value_ratio = pred["confidence"] * 100 / pred["starting_price"]
                print(
                    f"   {i}. {pred['greyhound_name']} (Box {pred['box']}) - {pred['confidence']*100:.1f}% @ ${pred['starting_price']:.1f}"
                )

            print(f"\n🎯 ADVANTAGES OF ENHANCED SYSTEM:")
            print("  • Always uses the most recent analysis data")
            print("  • Automatically copies files to analysis dataset")
            print("  • Provides confidence-weighted predictions")
            print("  • Adapts to all track venues automatically")
            print("  • Future-proof as analysis system evolves")

        else:
            print("❌ No predictions generated")

    except Exception as e:
        print(f"❌ Error running prediction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
