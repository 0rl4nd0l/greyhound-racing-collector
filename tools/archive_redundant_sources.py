
import os
import shutil
import argparse
from datetime import datetime

# Define the archive directory for legacy ingestion scripts
LEGACY_INGESTION_DIR = os.path.abspath("./archive/ingestion_legacy")

# Define criteria for identifying legacy scripts
# This is a list of script filenames that are considered outdated.
LEGACY_SCRIPTS = [
    "outdated_scripts/form_guide_scraper_2025.py",
    "outdated_scripts/greyhound_results_scraper_navigator.py",
    "archive/scripts_2025_07_23/advanced_scraper.py",
    "archive/scripts_2025_07_23/greyhound_results_scraper_navigator.py",
    "archive/test_scraper.py",
    "archive_unused_scripts/data_processing/enhanced_expert_form_scraper.py"
]

def archive_legacy_scripts(execute=False):
    """
    Archives legacy ingestion scripts to a dedicated directory.
    """
    if execute and not os.path.exists(LEGACY_INGESTION_DIR):
        print(f"Creating archive directory: {LEGACY_INGESTION_DIR}")
        os.makedirs(LEGACY_INGESTION_DIR)
        
    archived_files = []
    for script_path in LEGACY_SCRIPTS:
        if os.path.exists(script_path):
            destination_path = os.path.join(LEGACY_INGESTION_DIR, os.path.basename(script_path))
            
            # To ensure reversibility, we'll rename if a file with the same name exists
            if os.path.exists(destination_path):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                base, ext = os.path.splitext(os.path.basename(script_path))
                destination_path = os.path.join(LEGACY_INGESTION_DIR, f"{base}_{timestamp}{ext}")

            print(f"Found legacy script: {script_path}")
            if execute:
                print(f"  -> Archiving to: {destination_path}")
                shutil.move(script_path, destination_path)
                archived_files.append((script_path, destination_path))
            else:
                print(f"  -> (Dry Run) Would archive to: {destination_path}")
    
    if execute:
        print(f"\nSuccessfully archived {len(archived_files)} scripts.")
        # Create a manifest for reversibility
        manifest_path = os.path.join(LEGACY_INGESTION_DIR, "archive_manifest.txt")
        with open(manifest_path, "a") as f:
            for original, new in archived_files:
                f.write(f"{new} -> {original}\n")
        print(f"Manifest for reversal created at: {manifest_path}")


def reverse_archiving(manifest_path):
    """
    Reverses the archiving process using a manifest file.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if " -> " in line:
            new_path, original_path = line.strip().split(" -> ")
            if os.path.exists(new_path):
                print(f"Reversing: {new_path} -> {original_path}")
                # Ensure parent directory of original path exists
                os.makedirs(os.path.dirname(original_path), exist_ok=True)
                shutil.move(new_path, original_path)
    
    # Clean up the manifest after reversal
    os.remove(manifest_path)
    print("Reversal complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive legacy data ingestion scripts.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the files. Without this flag, it's a dry run.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the last archival operation.",
    )
    args = parser.parse_args()

    if args.reverse:
        manifest_file = os.path.join(LEGACY_INGESTION_DIR, "archive_manifest.txt")
        reverse_archiving(manifest_file)
    else:
        archive_legacy_scripts(execute=args.execute)

