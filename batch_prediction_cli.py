#!/usr/bin/env python3
"""
Batch Prediction CLI
====================

Command-line interface for the robust batch prediction pipeline.
Supports progress tracking, configuration options, and comprehensive error handling.

Usage:
    python batch_prediction_cli.py --input /path/to/csvs --output /path/to/results
    python batch_prediction_cli.py --input /path/to/csvs --output /path/to/results --workers 4 --chunk-size 500
    python batch_prediction_cli.py --job-status JOB_ID
    python batch_prediction_cli.py --cancel-job JOB_ID

Author: AI Assistant
Date: January 2025
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from batch_prediction_pipeline import (BatchPredictionPipeline,
                                     BatchProcessingConfig, create_batch_pipeline)


def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Robust Batch Prediction Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch prediction
  python batch_prediction_cli.py --input ./upcoming_races --output ./batch_results

  # Advanced options with custom configuration
  python batch_prediction_cli.py --input ./data --output ./results \\
    --workers 4 --chunk-size 500 --timeout 600 --validation strict

  # Check job status
  python batch_prediction_cli.py --job-status abc123def

  # Cancel running job
  python batch_prediction_cli.py --cancel-job abc123def

  # List all jobs
  python batch_prediction_cli.py --list-jobs
        """
    )
    
    # Main operation arguments
    parser.add_argument(
        '--input', '-i',
        help='Input directory containing CSV files',
        type=str
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for results',
        type=str
    )
    
    # Configuration arguments
    parser.add_argument(
        '--workers', '-w',
        help='Number of parallel workers (default: 3)',
        type=int,
        default=3
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        help='Batch size for processing (default: 10)',
        type=int,
        default=10
    )
    
    parser.add_argument(
        '--chunk-size', '-c',
        help='Chunk size for streaming large files (default: 1000)',
        type=int,
        default=1000
    )
    
    parser.add_argument(
        '--timeout', '-t',
        help='Timeout per file in seconds (default: 300)',
        type=int,
        default=300
    )
    
    parser.add_argument(
        '--validation', '-v',
        help='Validation level: strict, moderate, lenient (default: moderate)',
        choices=['strict', 'moderate', 'lenient'],
        default='moderate'
    )
    
    parser.add_argument(
        '--memory-limit', '-m',
        help='Memory limit per worker in MB (default: 512)',
        type=int,
        default=512
    )
    
    parser.add_argument(
        '--retry-attempts', '-r',
        help='Number of retry attempts for failed files (default: 3)',
        type=int,
        default=3
    )
    
    parser.add_argument(
        '--save-intermediate',
        help='Save intermediate results',
        action='store_true',
        default=True
    )
    
    parser.add_argument(
        '--cleanup-on-error',
        help='Cleanup files on error',
        action='store_true',
        default=False
    )
    
    # Job management arguments
    parser.add_argument(
        '--job-status',
        help='Check status of a specific job by ID',
        type=str
    )
    
    parser.add_argument(
        '--cancel-job',
        help='Cancel a running job by ID',
        type=str
    )
    
    parser.add_argument(
        '--list-jobs',
        help='List all batch jobs',
        action='store_true'
    )
    
    # Display options
    parser.add_argument(
        '--verbose', '-V',
        help='Enable verbose output',
        action='store_true'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        help='Suppress progress output',
        action='store_true'
    )
    
    parser.add_argument(
        '--format',
        help='Output format: text, json (default: text)',
        choices=['text', 'json'],
        default='text'
    )
    
    return parser


def progress_callback(job):
    """Progress callback for real-time updates"""
    if not args.quiet:
        progress_bar = "█" * int(job.progress / 5) + "░" * (20 - int(job.progress / 5))
        print(f"\r🚀 Job {job.job_id[:8]}: [{progress_bar}] {job.progress:.1f}% "
              f"({job.completed_files}/{job.total_files} files)", end="", flush=True)


def find_csv_files(input_dir):
    """Find all CSV files in the input directory"""
    csv_files = []
    input_path = Path(input_dir)
    
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        csv_files.append(str(input_path))
    elif input_path.is_dir():
        for file_path in input_path.rglob('*.csv'):
            csv_files.append(str(file_path))
    
    return csv_files


def format_output(data, format_type='text'):
    """Format output based on specified format"""
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    
    # Text format for various data types
    if isinstance(data, dict):
        if 'job_id' in data:  # Job status
            return format_job_status(data)
        else:
            return '\n'.join(f"{k}: {v}" for k, v in data.items())
    
    return str(data)


def format_job_status(job_data):
    """Format job status for text output"""
    status_icons = {
        'pending': '⏳',
        'running': '🔄',
        'completed': '✅',
        'completed_with_errors': '⚠️',
        'failed': '❌',
        'cancelled': '🛑'
    }
    
    icon = status_icons.get(job_data.get('status', 'unknown'), '❓')
    
    output = f"""
{icon} Job Status: {job_data.get('job_id', 'Unknown')[:12]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {job_data.get('name', 'N/A')}
Status: {job_data.get('status', 'Unknown')}
Progress: {job_data.get('progress', 0):.1f}%
Files: {job_data.get('completed_files', 0)}/{job_data.get('total_files', 0)}
Failed: {job_data.get('failed_files', 0)}
Created: {job_data.get('created_at', 'Unknown')}
Output Dir: {job_data.get('output_dir', 'N/A')}
"""
    
    if job_data.get('error_messages'):
        output += f"\nRecent Errors:\n"
        for error in job_data.get('error_messages', [])[-3:]:  # Last 3 errors
            output += f"  • {error}\n"
    
    return output


def main():
    global args
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.input, args.job_status, args.cancel_job, args.list_jobs]):
        parser.error("Must specify --input for batch processing, or use job management options")
    
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    try:
        # Initialize pipeline with configuration
        config = BatchProcessingConfig(
            chunk_size=args.chunk_size,
            max_memory_mb=args.memory_limit,
            timeout_seconds=args.timeout,
            retry_attempts=args.retry_attempts,
            validation_level=args.validation,
            save_intermediate=args.save_intermediate,
            cleanup_on_error=args.cleanup_on_error,
            progress_callback=progress_callback if not args.quiet else None
        )
        
        pipeline = create_batch_pipeline(config)
        
        # Handle job management operations
        if args.job_status:
            job = pipeline.get_job_status(args.job_status)
            if job:
                output = format_output(job.__dict__, args.format)
                print(output)
            else:
                print(f"❌ Job {args.job_status} not found")
                sys.exit(1)
            return
        
        if args.cancel_job:
            success = pipeline.cancel_job(args.cancel_job)
            if success:
                print(f"✅ Job {args.cancel_job} cancelled successfully")
            else:
                print(f"❌ Failed to cancel job {args.cancel_job}")
                sys.exit(1)
            return
        
        if args.list_jobs:
            jobs = pipeline.list_jobs()
            if args.format == 'json':
                output = json.dumps([job.__dict__ for job in jobs], indent=2, default=str)
                print(output)
            else:
                if not jobs:
                    print("📋 No batch jobs found")
                else:
                    print("📋 Batch Jobs:")
                    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    for job in jobs:
                        status_icon = {'pending': '⏳', 'running': '🔄', 'completed': '✅', 
                                     'failed': '❌', 'cancelled': '🛑'}.get(job.status, '❓')
                        print(f"{status_icon} {job.job_id[:12]} | {job.name} | {job.status} | "
                              f"{job.completed_files}/{job.total_files} files")
            return
        
        # Main batch processing
        if args.input:
            # Find CSV files
            csv_files = find_csv_files(args.input)
            
            if not csv_files:
                print(f"❌ No CSV files found in {args.input}")
                sys.exit(1)
            
            if args.verbose:
                print(f"🔍 Found {len(csv_files)} CSV files:")
                for i, file_path in enumerate(csv_files[:10], 1):  # Show first 10
                    print(f"  {i}. {Path(file_path).name}")
                if len(csv_files) > 10:
                    print(f"  ... and {len(csv_files) - 10} more files")
            
            # Create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Create batch job
            job_name = f"CLI Batch {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            if not args.quiet:
                print(f"🚀 Creating batch job: {job_name}")
                print(f"📂 Input: {args.input}")
                print(f"📁 Output: {args.output}")
                print(f"👥 Workers: {args.workers}")
                print(f"📦 Batch size: {args.batch_size}")
                print(f"🧩 Chunk size: {args.chunk_size}")
                print()
            
            job_id = pipeline.create_batch_job(
                name=job_name,
                input_files=csv_files,
                output_dir=args.output,
                batch_size=args.batch_size,
                max_workers=args.workers
            )
            
            if not args.quiet:
                print(f"📋 Job ID: {job_id}")
                print("🎯 Starting batch processing...")
                print()
            
            # Run batch job
            start_time = time.time()
            job = pipeline.run_batch_job(job_id)
            end_time = time.time()
            
            if not args.quiet:
                print()  # New line after progress bar
            
            # Display results
            duration = end_time - start_time
            status_icon = {'completed': '✅', 'failed': '❌', 'cancelled': '🛑', 
                          'completed_with_errors': '⚠️'}.get(job.status, '❓')
            
            if args.format == 'json':
                result = {
                    'job_id': job_id,
                    'status': job.status,
                    'duration_seconds': duration,
                    'total_files': job.total_files,
                    'completed_files': job.completed_files,
                    'failed_files': job.failed_files,
                    'error_messages': job.error_messages
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"{status_icon} Batch Processing Complete!")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"📋 Job ID: {job_id}")
                print(f"⏱️  Duration: {duration:.2f} seconds")
                print(f"📊 Status: {job.status}")
                print(f"✅ Completed: {job.completed_files}/{job.total_files} files")
                if job.failed_files > 0:
                    print(f"❌ Failed: {job.failed_files} files")
                print(f"📁 Results saved to: {args.output}")
                
                if job.error_messages and args.verbose:
                    print("\nRecent Errors:")
                    for error in job.error_messages[-5:]:  # Last 5 errors
                        print(f"  • {error}")
            
            # Set exit code based on job status
            if job.status == 'completed':
                sys.exit(0)
            elif job.status == 'completed_with_errors':
                sys.exit(2)  # Partial success
            else:
                sys.exit(1)  # Failure
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
