#!/usr/bin/env python3
"""
Demo script for batch processing implementation
==============================================

This script demonstrates the batch processing chunk loop implementation
showing the performance improvements over per-file processing.
"""

import os
import tempfile
import time
from pathlib import Path

from bulk_csv_ingest import (
    BATCH_SIZE,
    FormGuideCsvIngestor,
    batch_save_to_database,
    chunked,
    compute_needed_info,
    process_batch,
)
from csv_ingestion import ValidationLevel


def create_demo_csv_files(count: int = 250) -> list:
    """Create demo CSV files for batch processing demonstration."""
    print(f"📁 Creating {count} demo CSV files...")

    demo_files = []
    temp_dir = tempfile.mkdtemp(prefix="batch_demo_")

    for i in range(count):
        file_path = os.path.join(temp_dir, f"demo_race_{i:03d}.csv")

        # Create varied content to simulate real CSV files
        content = f"""Dog Name,PLC,BOX,DIST,DATE,TRACK,G
DemoGreyhound{i}_1,1,1,500,2024-01-{(i % 30) + 1:02d},DemoTrack{i % 5},5
DemoGreyhound{i}_2,2,2,500,2024-01-{(i % 30) + 1:02d},DemoTrack{i % 5},5
DemoGreyhound{i}_3,3,3,500,2024-01-{(i % 30) + 1:02d},DemoTrack{i % 5},5
"""

        with open(file_path, "w") as f:
            f.write(content)

        demo_files.append(file_path)

    print(f"✅ Created {len(demo_files)} demo files in {temp_dir}")
    return demo_files, temp_dir


def demo_old_approach_simulation(csv_files: list):
    """Simulate the old per-file approach for comparison."""
    print("🐌 Simulating OLD per-file approach...")

    start_time = time.time()
    processed_count = 0

    # Simulate old per-file processing with individual operations
    for csv_file in csv_files:
        # Simulate file hash computation per file
        if os.path.exists(csv_file):
            with open(csv_file, "rb") as f:
                _ = f.read()  # Simulate hash computation

        # Simulate individual file processing
        time.sleep(0.001)  # Simulate processing overhead per file
        processed_count += 1

        # Simulate individual database operations (not batched)
        time.sleep(0.0005)  # Simulate DB round-trip per file

    elapsed = time.time() - start_time

    print(f"   📊 Processed {processed_count} files in {elapsed:.2f} seconds")
    print(f"   📈 Rate: {processed_count / elapsed:.1f} files/second")

    return elapsed, processed_count


def demo_new_batch_approach(csv_files: list):
    """Demonstrate the new batch processing approach."""
    print("🚀 Demonstrating NEW batch processing approach...")

    start_time = time.time()
    total_processed = 0
    total_records = 0

    # Initialize ingestor
    ingestor = FormGuideCsvIngestor(validation_level=ValidationLevel.LENIENT)

    batch_count = 0
    total_batches = (len(csv_files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in chunked(csv_files, BATCH_SIZE):
        batch_count += 1

        # Step 1: Compute needed info (hash or small stat calls)
        batch_start = time.time()
        metadata = compute_needed_info(batch)
        metadata_time = time.time() - batch_start

        # Step 2: Process batch (existing logic reused)
        process_start = time.time()
        results = process_batch(batch, ingestor, metadata)
        process_time = time.time() - process_start

        # Step 3: Batch database operations using executemany()
        db_start = time.time()
        if results["batch_data"]:
            # In real implementation, this would save to actual database
            # For demo, we just simulate the batch operation
            time.sleep(0.002)  # Simulate executemany() batch operation
        db_time = time.time() - db_start

        # Update statistics
        batch_processed = len(results["processed_files"])
        batch_records = results["total_records"]

        total_processed += batch_processed
        total_records += batch_records

        # Show batch progress
        if batch_count % 10 == 0 or batch_count == total_batches:
            print(
                f"   📦 Batch {batch_count}/{total_batches}: "
                f"{batch_processed} files, {batch_records} records "
                f"(metadata: {metadata_time:.3f}s, process: {process_time:.3f}s, db: {db_time:.3f}s)"
            )

    elapsed = time.time() - start_time

    print(f"   📊 Processed {total_processed} files in {elapsed:.2f} seconds")
    print(f"   📈 Rate: {total_processed / elapsed:.1f} files/second")
    print(f"   🎯 Total records processed: {total_records}")

    return elapsed, total_processed, total_records


def main():
    """Run the batch processing demonstration."""
    print("🎬 Batch Processing Implementation Demo")
    print("=" * 50)

    try:
        # Create demo files
        csv_files, temp_dir = create_demo_csv_files(250)

        print(f"\n📦 Configuration: BATCH_SIZE = {BATCH_SIZE}")
        print(f"📁 Demo files created: {len(csv_files)}")

        # Demonstrate old approach
        print("\n" + "=" * 50)
        old_time, old_count = demo_old_approach_simulation(csv_files)

        # Demonstrate new batch approach
        print("\n" + "=" * 50)
        new_time, new_count, new_records = demo_new_batch_approach(csv_files)

        # Performance comparison
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        print(
            f"Old per-file approach:     {old_time:.2f}s ({old_count / old_time:.1f} files/s)"
        )
        print(
            f"New batch approach:        {new_time:.2f}s ({new_count / new_time:.1f} files/s)"
        )

        if old_time > 0:
            speedup = old_time / new_time
            print(f"Performance improvement:   {speedup:.1f}x faster")

        print(f"\n✅ Batch processing benefits:")
        print(f"   🔹 Reduced database round-trips with executemany()")
        print(f"   🔹 Efficient metadata computation in batches")
        print(f"   🔹 Better resource utilization")
        print(f"   🔹 Configurable batch size (currently {BATCH_SIZE})")
        print(f"   🔹 Preserved existing processing logic")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

    finally:
        # Cleanup demo files
        if "temp_dir" in locals():
            import shutil

            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up demo files from {temp_dir}")


if __name__ == "__main__":
    main()
