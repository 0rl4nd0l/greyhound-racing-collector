#!/usr/bin/env python3
"""
CLI Batch Predictor
===================

Command-line interface for running batch predictions with real-time progress tracking.
Supports both single file and batch processing with comprehensive error handling.

Usage:
    python cli_batch_predictor.py --file path/to/race.csv
    python cli_batch_predictor.py --batch path/to/directory/
    python cli_batch_predictor.py --upcoming-races
    python cli_batch_predictor.py --job-status <job_id>
    python cli_batch_predictor.py --list-jobs

Author: AI Assistant
Date: December 2024
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from batch_prediction_pipeline import BatchPredictionPipeline
    BATCH_PIPELINE_AVAILABLE = True
except ImportError:
    print("❌ Batch prediction pipeline not available. Please ensure batch_prediction_pipeline.py exists.")
    BATCH_PIPELINE_AVAILABLE = False

def print_banner():
    """Print CLI banner"""
    print("=" * 60)
    print("🎯 GREYHOUND RACING BATCH PREDICTOR CLI")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_progress_bar(progress, total, prefix="Progress", suffix="Complete", length=40):
    """Print a progress bar"""
    percent = ("{0:.1f}").format(100 * (progress / float(total)))
    filled_length = int(length * progress // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    if progress == total:
        print()  # New line when complete

def monitor_job_progress(pipeline, job_id, show_details=True):
    """Monitor job progress with real-time updates"""
    print(f"📊 Monitoring job: {job_id}")
    print("Press Ctrl+C to stop monitoring (job will continue in background)")
    print("-" * 60)
    
    last_progress = -1
    last_completed = -1
    
    try:
        while True:
            job = pipeline.get_job_status(job_id)
            
            if not job:
                print(f"❌ Job {job_id} not found")
                break
            
            # Update progress if changed
            if job.progress != last_progress or job.completed_files != last_completed:
                print_progress_bar(job.completed_files, job.total_files, 
                                 f"Job {job_id[:8]}", f"{job.completed_files}/{job.total_files} files")
                
                if show_details and job.completed_files != last_completed:
                    print(f"  Status: {job.status} | Progress: {job.progress:.1f}%")
                    if hasattr(job, 'current_file') and job.current_file:
                        print(f"  Current: {os.path.basename(job.current_file)}")
                    if job.failed_files > 0:
                        print(f"  ⚠️  Failed: {job.failed_files} files")
                
                last_progress = job.progress
                last_completed = job.completed_files
            
            # Check if job is complete
            if job.status in ['completed', 'failed', 'cancelled']:
                print()  # New line
                print(f"🏁 Job {job_id[:8]} finished with status: {job.status}")
                print(f"📈 Results: {job.completed_files} completed, {job.failed_files} failed")
                
                # Show error summary if there were failures
                if job.failed_files > 0 and job.error_messages:
                    print("\n❌ Recent errors:")
                    for error in job.error_messages[-3:]:  # Last 3 errors
                        print(f"   • {error}")
                
                break
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Monitoring stopped. Job {job_id[:8]} continues in background.")
        print(f"   Check status with: python {sys.argv[0]} --job-status {job_id}")

def predict_single_file(file_path, pipeline=None, historical=False):
    """Predict a single race file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    if not BATCH_PIPELINE_AVAILABLE:
        print("❌ Batch prediction pipeline not available")
        return False
    
    if not pipeline:
        pipeline = BatchPredictionPipeline()
    
    print(f"🎯 Predicting single file: {os.path.basename(file_path)}")
    
    try:
        # Create a single-file batch job
        job_id = pipeline.create_batch_job(
            name=f"CLI Single File: {os.path.basename(file_path)}",
            input_files=[file_path],
            output_dir="./cli_predictions",
            batch_size=1,
            max_workers=1,
            historical=historical
        )
        
        print(f"📋 Created job: {job_id}")
        
        # Run the job
        pipeline.run_batch_job(job_id)
        
        # Monitor progress
        monitor_job_progress(pipeline, job_id, show_details=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error predicting file: {e}")
        return False

def predict_batch_directory(directory_path, pipeline=None, historical=False):
    """Predict all CSV files in a directory"""
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return False
    
    if not BATCH_PIPELINE_AVAILABLE:
        print("❌ Batch prediction pipeline not available")
        return False
    
    if not pipeline:
        pipeline = BatchPredictionPipeline()
    
    # Find all CSV files
    csv_files = []
    for file_path in Path(directory_path).glob("*.csv"):
        csv_files.append(str(file_path))
    
    if not csv_files:
        print(f"❌ No CSV files found in: {directory_path}")
        return False
    
    print(f"📁 Found {len(csv_files)} CSV files in: {directory_path}")
    
    try:
        # Create batch job
        job_id = pipeline.create_batch_job(
            name=f"CLI Batch: {os.path.basename(directory_path)}",
            input_files=csv_files,
            output_dir="./cli_batch_predictions",
            batch_size=10,
            max_workers=3,
            historical=historical
        )
        
        print(f"📋 Created batch job: {job_id}")
        
        # Run the job in background
        import threading
        job_thread = threading.Thread(target=pipeline.run_batch_job, args=(job_id,), daemon=True)
        job_thread.start()
        
        # Monitor progress
        monitor_job_progress(pipeline, job_id, show_details=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing batch: {e}")
        return False

def predict_upcoming_races(pipeline=None):
    """Predict all upcoming races"""
    upcoming_dir = "./upcoming_races"
    
    if not os.path.exists(upcoming_dir):
        print(f"❌ Upcoming races directory not found: {upcoming_dir}")
        return False
    
    return predict_batch_directory(upcoming_dir, pipeline)

def show_job_status(job_id, pipeline=None):
    """Show detailed job status"""
    if not BATCH_PIPELINE_AVAILABLE:
        print("❌ Batch prediction pipeline not available")
        return False
    
    if not pipeline:
        pipeline = BatchPredictionPipeline()
    
    job = pipeline.get_job_status(job_id)
    
    if not job:
        print(f"❌ Job not found: {job_id}")
        return False
    
    print(f"📊 Job Status: {job_id}")
    print("-" * 40)
    print(f"Name: {job.name}")
    print(f"Status: {job.status}")
    print(f"Progress: {job.progress:.1f}%")
    print(f"Files: {job.completed_files}/{job.total_files} (Failed: {job.failed_files})")
    print(f"Created: {job.created_at}")
    
    if hasattr(job, 'current_file') and job.current_file:
        print(f"Current: {os.path.basename(job.current_file)}")
    
    if job.error_messages:
        print("\nRecent Errors:")
        for error in job.error_messages[-5:]:
            print(f"  • {error}")
    
    # If job is running, offer to monitor
    if job.status == 'running':
        response = input("\nMonitor progress? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            monitor_job_progress(pipeline, job_id)
    
    return True

def list_jobs(pipeline=None):
    """List all batch jobs"""
    if not BATCH_PIPELINE_AVAILABLE:
        print("❌ Batch prediction pipeline not available")
        return False
    
    if not pipeline:
        pipeline = BatchPredictionPipeline()
    
    jobs = pipeline.list_jobs()
    
    if not jobs:
        print("📋 No batch jobs found")
        return True
    
    print(f"📋 Found {len(jobs)} batch jobs:")
    print("-" * 80)
    print(f"{'Job ID':<12} {'Name':<25} {'Status':<12} {'Progress':<10} {'Files':<10}")
    print("-" * 80)
    
    for job in jobs:
        job_id_short = job.job_id[:8] if len(job.job_id) > 8 else job.job_id
        name_short = job.name[:23] + "..." if len(job.name) > 25 else job.name
        files_info = f"{job.completed_files}/{job.total_files}"
        
        print(f"{job_id_short:<12} {name_short:<25} {job.status:<12} {job.progress:<10.1f}% {files_info:<10}")
    
    return True

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="CLI Batch Predictor for Greyhound Racing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file race.csv                    # Predict single file with required headers
  %(prog)s --batch ./upcoming_races/         # Predict all files with manifest behavior
  %(prog)s --upcoming-races                  # Predict with debug mode enabled
  %(prog)s --job-status abc123               # Check job status
  %(prog)s --list-jobs                       # List all jobs
"""
    )
    
    parser.add_argument('--file', '-f', 
                       help='Predict a single CSV file')
    parser.add_argument('--batch', '-b', 
                       help='Predict all CSV files in directory')
    parser.add_argument('--upcoming-races', '-u', action='store_true',
                       help='Predict all upcoming races')
    parser.add_argument('--job-status', '-s', 
                       help='Show status of specific job')
    parser.add_argument('--list-jobs', '-l', action='store_true',
                       help='List all batch jobs')
    parser.add_argument('--progress-callback', action='store_true',
                       help='Enable detailed progress callbacks (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--historical', action='store_true',
                       help='Enable historical mode - only process races with dates < today')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if not any([args.file, args.batch, args.upcoming_races, args.job_status, args.list_jobs]):
        parser.print_help()
        sys.exit(1)
    
    if not args.quiet:
        print_banner()
    
    # Initialize pipeline once
    pipeline = None
    if BATCH_PIPELINE_AVAILABLE:
        try:
            pipeline = BatchPredictionPipeline()
        except Exception as e:
            print(f"❌ Failed to initialize batch pipeline: {e}")
            sys.exit(1)
    
    success = True
    
    try:
        if args.file:
            success = predict_single_file(args.file, pipeline, args.historical)
        
        elif args.batch:
            success = predict_batch_directory(args.batch, pipeline, args.historical)
        
        elif args.upcoming_races:
            success = predict_upcoming_races(pipeline)
        
        elif args.job_status:
            success = show_job_status(args.job_status, pipeline)
        
        elif args.list_jobs:
            success = list_jobs(pipeline)
    
    except KeyboardInterrupt:
        print("\n⏸️  Operation cancelled by user")
        success = False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        success = False
    
    if not args.quiet:
        print()
        if success:
            print("✅ CLI operation completed successfully")
        else:
            print("❌ CLI operation failed")
        print("=" * 60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
