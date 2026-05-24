import os
import shutil

# Add the tools directory to the Python path to import the script
import sys
import unittest
from unittest.mock import patch

tools_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
sys.path.append(tools_path)

from archive_redundant_sources import (
    LEGACY_SCRIPTS,
    archive_legacy_scripts,
    reverse_archiving,
)


class TestArchiveRedundantSources(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for testing."""
        self.test_dir = os.path.abspath("./test_temp_archive")
        self.archive_dir = os.path.join(self.test_dir, "archive/ingestion_legacy")

        # Create dummy legacy scripts for testing
        for script_path in LEGACY_SCRIPTS:
            # Create the full path for the dummy script in the test directory
            full_path = os.path.join(self.test_dir, script_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(f"# Test script: {os.path.basename(full_path)}")

        # Patch the constants to use the test directory
        self.legacy_ingestion_dir_patch = patch(
            "archive_redundant_sources.LEGACY_INGESTION_DIR", self.archive_dir
        )
        self.legacy_scripts_patch = patch(
            "archive_redundant_sources.LEGACY_SCRIPTS",
            [os.path.join(self.test_dir, p) for p in LEGACY_SCRIPTS],
        )

        self.legacy_ingestion_dir_patch.start()
        self.legacy_scripts_patch.start()

    def tearDown(self):
        """Clean up the temporary environment after testing."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        self.legacy_ingestion_dir_patch.stop()
        self.legacy_scripts_patch.stop()

    def test_dry_run_does_not_move_files(self):
        """Verify that a dry run identifies files but does not move them."""
        # Get the original paths of the dummy scripts
        original_script_paths = [os.path.join(self.test_dir, p) for p in LEGACY_SCRIPTS]

        # Run the archive script in dry-run mode
        archive_legacy_scripts(execute=False)

        # Check that the archive directory was not created
        self.assertFalse(os.path.exists(self.archive_dir))

        # Check that the original files still exist
        for script_path in original_script_paths:
            self.assertTrue(os.path.exists(script_path))

    def test_execute_moves_files_and_creates_manifest(self):
        """Verify that executing the script moves files and creates a manifest."""
        original_script_paths = [os.path.join(self.test_dir, p) for p in LEGACY_SCRIPTS]

        # Run the archive script with execute=True
        archive_legacy_scripts(execute=True)

        # Check that the archive directory was created
        self.assertTrue(os.path.exists(self.archive_dir))

        # Check that the original files have been moved
        for script_path in original_script_paths:
            self.assertFalse(os.path.exists(script_path))
            # Check that the file exists in the archive
            archived_path = os.path.join(
                self.archive_dir, os.path.basename(script_path)
            )
            self.assertTrue(os.path.exists(archived_path))

        # Check that the manifest file was created
        manifest_path = os.path.join(self.archive_dir, "archive_manifest.txt")
        self.assertTrue(os.path.exists(manifest_path))

        # Check the manifest content
        with open(manifest_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(LEGACY_SCRIPTS))

    def test_reversal_restores_files(self):
        """Verify that the reversal process restores files to their original locations."""
        original_script_paths = [os.path.join(self.test_dir, p) for p in LEGACY_SCRIPTS]
        manifest_path = os.path.join(self.archive_dir, "archive_manifest.txt")

        # First, run the archival
        archive_legacy_scripts(execute=True)
        self.assertTrue(os.path.exists(manifest_path))

        # Now, run the reversal
        reverse_archiving(manifest_path)

        # Check that the original files are restored
        for script_path in original_script_paths:
            self.assertTrue(os.path.exists(script_path))

        # Check that the manifest file is removed after reversal
        self.assertFalse(os.path.exists(manifest_path))

    def test_conflict_resolution(self):
        """Verify that the script handles filename conflicts by appending a timestamp."""
        # Create a conflicting file in the archive directory beforehand
        os.makedirs(self.archive_dir, exist_ok=True)
        conflicting_script_path = os.path.join(self.test_dir, LEGACY_SCRIPTS[0])
        archive_conflict_path = os.path.join(
            self.archive_dir, os.path.basename(conflicting_script_path)
        )
        with open(archive_conflict_path, "w") as f:
            f.write("# Pre-existing file")

        # Run the archival
        archive_legacy_scripts(execute=True)

        # Check that the original file was moved and renamed
        self.assertFalse(os.path.exists(conflicting_script_path))

        # Find the new, timestamped file
        archived_files = os.listdir(self.archive_dir)
        base, ext = os.path.splitext(os.path.basename(conflicting_script_path))
        found_timestamped = False
        for f in archived_files:
            if (
                f.startswith(base)
                and f.endswith(ext)
                and f != os.path.basename(conflicting_script_path)
            ):
                found_timestamped = True
                break
        self.assertTrue(found_timestamped)


if __name__ == "__main__":
    unittest.main()
