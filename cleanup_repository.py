#!/usr/bin/env python3
"""
Repository Cleanup and Organization Script
Organizes loose scripts and archives outdated files according to project structure
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directory structure"""
    directories = [
        "tools",  # Utility scripts
        "scripts",  # Operational scripts
        "tests",  # Test files
        "archive/scripts",  # Outdated scripts
        "docs/archived",  # Archived documentation
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ensured: {directory}")


def get_file_categories():
    """Define file categorization rules"""
    return {
        "tools": {
            "patterns": [
                "fix_*.py",
                "check_*.py",
                "repair_*.py",
                "cleanup_*.py",
                "analyze_*.py",
                "validate_*.py",
                "assess_*.py",
            ],
            "description": "Utility and maintenance tools",
        },
        "scripts": {
            "patterns": [
                "create_*.py",
                "setup_*.py",
                "run_*.py",
                "start_*.py",
                "batch_*.py",
                "bulk_*.py",
                "daily_*.py",
            ],
            "description": "Operational scripts",
        },
        "tests": {
            "patterns": ["test_*.py", "*_test.py", "conftest.py", "*TESTING*.py"],
            "description": "Test files",
        },
        "archive": {
            "patterns": [
                "old_*.py",
                "legacy_*.py",
                "deprecated_*.py",
                "backup_*.py",
                "temp_*.py",
                "experimental_*.py",
                "*_old.py",
                "*_backup.py",
            ],
            "description": "Outdated or experimental files",
        },
    }


def should_archive_file(filename, content_sample=""):
    """Determine if a file should be archived based on various criteria"""
    archive_indicators = [
        # Filename patterns
        "old_",
        "legacy_",
        "deprecated_",
        "backup_",
        "temp_",
        "_old",
        "_backup",
        "_v1",
        "_v2",
        "experimental_",
        "test_old",
        "draft_",
        # Content-based (if we have sample)
        "TODO: Remove this",
        "DEPRECATED",
        "OBSOLETE",
    ]

    filename_lower = filename.lower()
    for indicator in archive_indicators:
        if indicator in filename_lower:
            return True

    return False


def categorize_python_files():
    """Categorize all Python files in root directory"""
    root_files = [f for f in os.listdir(".") if f.endswith(".py") and os.path.isfile(f)]
    categories = get_file_categories()

    results = {category: [] for category in categories}
    results["uncategorized"] = []

    for file in root_files:
        categorized = False

        # Check if file should be archived first
        if should_archive_file(file):
            results["archive"].append(file)
            categorized = True
        else:
            # Check other categories
            for category, rules in categories.items():
                if category == "archive":
                    continue

                for pattern in rules["patterns"]:
                    if file.startswith(pattern.replace("*", "")) or file.endswith(
                        pattern.replace("*", "")
                    ):
                        results[category].append(file)
                        categorized = True
                        break

                if categorized:
                    break

        if not categorized:
            results["uncategorized"].append(file)

    return results


def move_files_safely(file_list, target_dir, action="move"):
    """Move or copy files to target directory with safety checks"""
    moved_count = 0

    for file in file_list:
        if not os.path.exists(file):
            logger.warning(f"File not found: {file}")
            continue

        target_path = os.path.join(target_dir, file)

        # Don't overwrite existing files
        if os.path.exists(target_path):
            logger.warning(f"Target exists, skipping: {target_path}")
            continue

        try:
            if action == "move":
                shutil.move(file, target_path)
                logger.info(f"Moved: {file} → {target_path}")
            else:
                shutil.copy2(file, target_path)
                logger.info(f"Copied: {file} → {target_path}")
            moved_count += 1
        except Exception as e:
            logger.error(f"Failed to {action} {file}: {e}")

    return moved_count


def cleanup_repository():
    """Main cleanup function"""
    logger.info("🧹 Starting repository cleanup...")

    # Create directory structure
    create_directories()

    # Categorize files
    categorized = categorize_python_files()

    # Report what was found
    logger.info("📊 File categorization results:")
    total_files = 0
    for category, files in categorized.items():
        count = len(files)
        total_files += count
        if count > 0:
            logger.info(f"  {category}: {count} files")
            for file in files[:3]:  # Show first 3 files
                logger.info(f"    - {file}")
            if count > 3:
                logger.info(f"    ... and {count-3} more")

    # Move files to appropriate directories
    total_moved = 0

    # Move tools
    if categorized["tools"]:
        moved = move_files_safely(categorized["tools"], "tools")
        total_moved += moved

    # Move scripts
    if categorized["scripts"]:
        moved = move_files_safely(categorized["scripts"], "scripts")
        total_moved += moved

    # Move tests
    if categorized["tests"]:
        moved = move_files_safely(categorized["tests"], "tests")
        total_moved += moved

    # Archive old files
    if categorized["archive"]:
        archive_dir = (
            f"archive/scripts/cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        Path(archive_dir).mkdir(parents=True, exist_ok=True)
        moved = move_files_safely(categorized["archive"], archive_dir)
        total_moved += moved

    # Report uncategorized files
    if categorized["uncategorized"]:
        logger.info("⚠️ Uncategorized files (left in root):")
        for file in categorized["uncategorized"]:
            logger.info(f"  - {file}")

    logger.info(f"✅ Cleanup complete! Organized {total_moved}/{total_files} files")

    return total_moved


if __name__ == "__main__":
    moved_count = cleanup_repository()
    print(f"\\n🎯 Repository cleanup complete! Organized {moved_count} files.")
