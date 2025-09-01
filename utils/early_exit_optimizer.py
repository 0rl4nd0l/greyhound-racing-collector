#!/usr/bin/env python3
"""
Early-Exit Strategy for Mostly Cached Directories
=================================================

This module implements an early-exit optimization strategy for directories where
most files have already been processed (cached). When pre-filtering removes
≥95% of files AND the unprocessed count is below a threshold, it prints a
summary and returns immediately, skipping detailed progress printing that 
costs time in huge directories.

Features:
- Configurable cache ratio threshold (default: 95%)
- Configurable unprocessed file threshold (default: 5)
- Fast summary generation for mostly-cached directories
- Integration with existing caching and batch processing systems

Author: AI Assistant
Date: 2025-01-15
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logger import logger


@dataclass
class EarlyExitConfig:
    """Configuration for early-exit strategy"""

    cache_ratio_threshold: float = 0.95  # 95% cached threshold
    unprocessed_threshold: int = 5  # Max unprocessed files for early exit
    enable_early_exit: bool = True  # Master switch
    verbose_summary: bool = True  # Show detailed summary


@dataclass
class DirectoryScanResult:
    """Result of directory scanning with early-exit analysis"""

    total_files: int
    processed_files: int
    unprocessed_files: int
    cache_ratio: float
    should_early_exit: bool
    scan_duration: float
    directory_path: str


class EarlyExitOptimizer:
    """
    Implements early-exit strategy for mostly cached directories.

    This optimizer analyzes directories to determine if they are "mostly cached"
    and can benefit from early-exit optimization. When conditions are met,
    it skips detailed processing and returns a summary immediately.
    """

    def __init__(self, config: Optional[EarlyExitConfig] = None):
        """
        Initialize the early-exit optimizer.

        Args:
            config: Configuration for early-exit behavior
        """
        self.config = config or EarlyExitConfig()
        logger.debug(f"EarlyExitOptimizer initialized with config: {self.config}")

    def analyze_directory_cache_status(
        self,
        directory: str,
        processed_files_set: set,
        file_extensions: Optional[List[str]] = None,
    ) -> DirectoryScanResult:
        """
        Analyze directory to determine cache status and early-exit eligibility.

        Args:
            directory: Directory path to analyze
            processed_files_set: Set of already processed filenames for O(1) lookup
            file_extensions: File extensions to consider (default: ['.csv'])

        Returns:
            DirectoryScanResult with analysis details
        """
        start_time = time.time()

        if file_extensions is None:
            file_extensions = [".csv"]

        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return DirectoryScanResult(
                total_files=0,
                processed_files=0,
                unprocessed_files=0,
                cache_ratio=0.0,
                should_early_exit=False,
                scan_duration=time.time() - start_time,
                directory_path=directory,
            )

        total_files = 0
        processed_files = 0

        try:
            # Use os.scandir for efficient scanning
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue

                    # Check file extension
                    file_ext = Path(entry.name).suffix.lower()
                    if file_extensions and file_ext not in file_extensions:
                        continue

                    total_files += 1

                    # Check if file is already processed using O(1) set lookup
                    if entry.name in processed_files_set:
                        processed_files += 1

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return DirectoryScanResult(
                total_files=0,
                processed_files=0,
                unprocessed_files=0,
                cache_ratio=0.0,
                should_early_exit=False,
                scan_duration=time.time() - start_time,
                directory_path=directory,
            )

        unprocessed_files = total_files - processed_files
        cache_ratio = processed_files / total_files if total_files > 0 else 0.0

        # Determine if early exit should be triggered
        should_early_exit = (
            self.config.enable_early_exit
            and cache_ratio >= self.config.cache_ratio_threshold
            and unprocessed_files <= self.config.unprocessed_threshold
            and total_files > 0  # Don't early exit on empty directories
        )

        scan_duration = time.time() - start_time

        result = DirectoryScanResult(
            total_files=total_files,
            processed_files=processed_files,
            unprocessed_files=unprocessed_files,
            cache_ratio=cache_ratio,
            should_early_exit=should_early_exit,
            scan_duration=scan_duration,
            directory_path=directory,
        )

        logger.debug(f"Directory analysis complete: {result}")
        return result

    def print_early_exit_summary(self, scan_result: DirectoryScanResult):
        """
        Print a concise summary for early-exit directories.

        Args:
            scan_result: Result from directory analysis
        """
        if not self.config.verbose_summary:
            return

        cache_percentage = scan_result.cache_ratio * 100

        print(f"🚀 Early-exit optimization triggered for {scan_result.directory_path}")
        print(f"   📊 Total files: {scan_result.total_files:,}")
        print(
            f"   ✅ Cached/processed: {scan_result.processed_files:,} ({cache_percentage:.1f}%)"
        )
        print(f"   📝 Unprocessed: {scan_result.unprocessed_files}")
        print(f"   ⚡ Scan duration: {scan_result.scan_duration:.3f}s")
        print(f"   💡 Skipping detailed progress printing to optimize performance")

        if scan_result.unprocessed_files > 0:
            print(
                f"   🔄 Remaining {scan_result.unprocessed_files} files will be processed quickly"
            )

    def should_use_early_exit(
        self,
        directory: str,
        processed_files_set: set,
        file_extensions: Optional[List[str]] = None,
    ) -> Tuple[bool, DirectoryScanResult]:
        """
        Determine if early-exit strategy should be used for a directory.

        Args:
            directory: Directory path to check
            processed_files_set: Set of already processed filenames
            file_extensions: File extensions to consider

        Returns:
            Tuple of (should_use_early_exit, scan_result)
        """
        scan_result = self.analyze_directory_cache_status(
            directory, processed_files_set, file_extensions
        )

        if scan_result.should_early_exit:
            logger.info(
                f"Early-exit triggered for {directory}: "
                f"{scan_result.cache_ratio:.1%} cached, "
                f"{scan_result.unprocessed_files} unprocessed"
            )

            if self.config.verbose_summary:
                self.print_early_exit_summary(scan_result)

        return scan_result.should_early_exit, scan_result

    def get_unprocessed_files_fast(
        self,
        directory: str,
        processed_files_set: set,
        file_extensions: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Quickly get list of unprocessed files for early-exit scenarios.

        This method is optimized for the case where we know there are very few
        unprocessed files and we want to get them quickly.

        Args:
            directory: Directory to scan
            processed_files_set: Set of processed filenames
            file_extensions: File extensions to consider

        Returns:
            List of unprocessed file paths
        """
        if file_extensions is None:
            file_extensions = [".csv"]

        unprocessed_files = []

        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue

                    # Check file extension
                    file_ext = Path(entry.name).suffix.lower()
                    if file_extensions and file_ext not in file_extensions:
                        continue

                    # If not in processed set, it's unprocessed
                    if entry.name not in processed_files_set:
                        unprocessed_files.append(entry.path)

                        # Early termination if we exceed threshold
                        # (indicates directory shouldn't use early exit)
                        if len(unprocessed_files) > self.config.unprocessed_threshold:
                            break

        except Exception as e:
            logger.error(f"Error getting unprocessed files from {directory}: {e}")
            return []

        return unprocessed_files

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")


def create_early_exit_optimizer(
    cache_ratio_threshold: float = 0.95,
    unprocessed_threshold: int = 5,
    enable_early_exit: bool = True,
    verbose_summary: bool = True,
) -> EarlyExitOptimizer:
    """
    Factory function to create an EarlyExitOptimizer with custom configuration.

    Args:
        cache_ratio_threshold: Minimum cache ratio for early exit (0.0-1.0)
        unprocessed_threshold: Maximum unprocessed files for early exit
        enable_early_exit: Master switch for early exit functionality
        verbose_summary: Whether to print detailed summaries

    Returns:
        Configured EarlyExitOptimizer instance
    """
    config = EarlyExitConfig(
        cache_ratio_threshold=cache_ratio_threshold,
        unprocessed_threshold=unprocessed_threshold,
        enable_early_exit=enable_early_exit,
        verbose_summary=verbose_summary,
    )

    return EarlyExitOptimizer(config)


# Integration helper functions
def check_directory_for_early_exit(
    directory: str,
    processed_files_set: set,
    cache_ratio_threshold: float = 0.95,
    unprocessed_threshold: int = 5,
) -> Tuple[bool, Optional[List[str]]]:
    """
    Convenience function to check if a directory qualifies for early exit.

    Args:
        directory: Directory to check
        processed_files_set: Set of processed filenames
        cache_ratio_threshold: Cache ratio threshold for early exit
        unprocessed_threshold: Max unprocessed files for early exit

    Returns:
        Tuple of (should_early_exit, unprocessed_files_list)
        If should_early_exit is True, unprocessed_files_list contains the files to process
        If should_early_exit is False, unprocessed_files_list is None
    """
    optimizer = create_early_exit_optimizer(
        cache_ratio_threshold=cache_ratio_threshold,
        unprocessed_threshold=unprocessed_threshold,
    )

    should_exit, scan_result = optimizer.should_use_early_exit(
        directory, processed_files_set
    )

    if should_exit:
        # Get the small number of unprocessed files
        unprocessed_files = optimizer.get_unprocessed_files_fast(
            directory, processed_files_set
        )
        return True, unprocessed_files

    return False, None


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test early-exit optimization")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument(
        "--cache-ratio",
        type=float,
        default=0.95,
        help="Cache ratio threshold (default: 0.95)",
    )
    parser.add_argument(
        "--unprocessed-threshold",
        type=int,
        default=5,
        help="Unprocessed files threshold (default: 5)",
    )
    parser.add_argument(
        "--disable-early-exit",
        action="store_true",
        help="Disable early exit optimization",
    )

    args = parser.parse_args()

    # Create optimizer with custom config
    optimizer = create_early_exit_optimizer(
        cache_ratio_threshold=args.cache_ratio,
        unprocessed_threshold=args.unprocessed_threshold,
        enable_early_exit=not args.disable_early_exit,
    )

    # For testing, create a mock processed files set
    # In real usage, this would come from the caching utilities
    test_processed_files = {
        f"test_file_{i:03d}.csv" for i in range(100)  # Simulate 100 processed files
    }

    print(f"🧪 Testing early-exit optimization for: {args.directory}")
    print(f"📊 Configuration:")
    print(f"   Cache ratio threshold: {args.cache_ratio:.1%}")
    print(f"   Unprocessed threshold: {args.unprocessed_threshold}")
    print(f"   Early exit enabled: {not args.disable_early_exit}")
    print()

    # Test the optimizer
    should_exit, scan_result = optimizer.should_use_early_exit(
        args.directory, test_processed_files
    )

    print(f"📈 Analysis Results:")
    print(f"   Total files: {scan_result.total_files}")
    print(f"   Processed files: {scan_result.processed_files}")
    print(f"   Unprocessed files: {scan_result.unprocessed_files}")
    print(f"   Cache ratio: {scan_result.cache_ratio:.1%}")
    print(f"   Should early exit: {should_exit}")
    print(f"   Scan duration: {scan_result.scan_duration:.3f}s")

    if should_exit:
        print("\n✅ Early exit optimization would be triggered!")
        unprocessed_files = optimizer.get_unprocessed_files_fast(
            args.directory, test_processed_files
        )
        print(f"📝 Unprocessed files to handle: {len(unprocessed_files)}")
        for file_path in unprocessed_files[:10]:  # Show first 10
            print(f"   - {os.path.basename(file_path)}")
        if len(unprocessed_files) > 10:
            print(f"   ... and {len(unprocessed_files) - 10} more")
    else:
        print(
            "\n⏳ Directory does not qualify for early exit - normal processing required"
        )
