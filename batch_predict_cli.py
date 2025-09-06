#!/usr/bin/env python3
"""
Batch Prediction CLI
===================

Command-line interface for running batch predictions with progress tracking,
streaming output, and comprehensive error handling.

Features:
- Interactive CLI with progress bars
- Real-time status updates
- Job management (create, monitor, cancel)
- Flexible input/output handling
- Resume capabilities for interrupted jobs

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

try:
    from batch_prediction_pipeline import (
        BatchJob,
        BatchPredictionPipeline,
        BatchProcessingConfig,
        create_batch_pipeline,
    )
except ImportError:
    print("❌ Error: batch_prediction_pipeline module not found")
    sys.exit(1)

console = Console()


class BatchCLI:
    """CLI interface for batch prediction operations"""

    def __init__(self):
        self.pipeline: Optional[BatchPredictionPipeline] = None
        self.current_job_id: Optional[str] = None

    def setup_pipeline(self, config: BatchProcessingConfig):
        """Initialize the batch prediction pipeline"""
        try:
            self.pipeline = create_batch_pipeline(config)
            console.print("✅ Batch prediction pipeline initialized", style="green")
        except Exception as e:
            console.print(f"❌ Failed to initialize pipeline: {e}", style="red")
            sys.exit(1)

    def find_csv_files(self, input_path: str, recursive: bool = True) -> List[str]:
        """Find CSV files in the input path"""
        csv_files = []
        input_path = Path(input_path)

        if input_path.is_file() and input_path.suffix.lower() == ".csv":
            csv_files.append(str(input_path))
        elif input_path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            csv_files.extend([str(f) for f in input_path.glob(pattern)])

        return sorted(csv_files)

    def create_job_interactive(self) -> str:
        """Interactive job creation"""
        console.print("\n🚀 [bold blue]Create New Batch Prediction Job[/bold blue]")

        # Get job name
        job_name = click.prompt("Job name", default=f"batch_job_{int(time.time())}")

        # Get input path
        input_path = click.prompt("Input path (file or directory)")
        if not os.path.exists(input_path):
            console.print(f"❌ Input path does not exist: {input_path}", style="red")
            return None

        # Find CSV files
        recursive = click.confirm("Search recursively for CSV files?", default=True)
        csv_files = self.find_csv_files(input_path, recursive)

        if not csv_files:
            console.print(f"❌ No CSV files found in: {input_path}", style="red")
            return None

        console.print(f"📁 Found {len(csv_files)} CSV files")

        # Show first few files as preview
        if len(csv_files) <= 10:
            for f in csv_files:
                console.print(f"  - {os.path.basename(f)}")
        else:
            for f in csv_files[:5]:
                console.print(f"  - {os.path.basename(f)}")
            console.print(f"  ... and {len(csv_files) - 5} more files")

        if not click.confirm(f"Proceed with {len(csv_files)} files?"):
            return None

        # Get output directory
        output_dir = click.prompt("Output directory", default="./batch_output")

        # Get batch processing parameters
        batch_size = click.prompt("Batch size", default=10, type=int)
        max_workers = click.prompt("Max workers", default=3, type=int)

        try:
            job_id = self.pipeline.create_batch_job(
                name=job_name,
                input_files=csv_files,
                output_dir=output_dir,
                batch_size=batch_size,
                max_workers=max_workers,
            )

            console.print(f"✅ Created job: {job_id}", style="green")
            return job_id

        except Exception as e:
            console.print(f"❌ Failed to create job: {e}", style="red")
            return None

    def run_job_with_progress(self, job_id: str):
        """Run a job with real-time progress tracking"""
        if not self.pipeline:
            console.print("❌ Pipeline not initialized", style="red")
            return

        job = self.pipeline.get_job_status(job_id)
        if not job:
            console.print(f"❌ Job {job_id} not found", style="red")
            return

        console.print(f"\n🚀 [bold blue]Starting Job: {job.name}[/bold blue]")
        console.print(f"📂 Input files: {job.total_files}")
        console.print(f"📁 Output dir: {job.output_dir}")
        console.print(f"⚙️ Workers: {job.max_workers}")

        # Create progress tracking
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        # Create status table
        def create_status_table(job: BatchJob) -> Table:
            table = Table(title=f"Job Status: {job.name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Status", job.status.upper())
            table.add_row("Progress", f"{job.progress:.1f}%")
            table.add_row("Completed", str(job.completed_files))
            table.add_row("Failed", str(job.failed_files))
            table.add_row(
                "Remaining",
                str(job.total_files - job.completed_files - job.failed_files),
            )

            if job.error_messages:
                table.add_row("Recent Errors", str(len(job.error_messages)))

            return table

        # Start the job in a separate thread
        import threading

        job_thread = threading.Thread(
            target=self.pipeline.run_batch_job, args=(job_id,)
        )
        job_thread.daemon = True
        job_thread.start()

        # Track progress with live updates
        with Live(console=console, refresh_per_second=2) as live:
            with progress:
                task_id = progress.add_task("Processing files", total=job.total_files)

                while job_thread.is_alive() or job.status == "running":
                    # Update job status
                    current_job = self.pipeline.get_job_status(job_id)
                    if current_job:
                        job = current_job

                        # Update progress
                        completed = job.completed_files + job.failed_files
                        progress.update(task_id, completed=completed)

                        # Create live display
                        status_panel = Panel(
                            create_status_table(job),
                            title="Batch Prediction Status",
                            border_style="blue",
                        )

                        live.update(status_panel)

                        # Check if job completed
                        if job.status in [
                            "completed",
                            "completed_with_errors",
                            "failed",
                            "cancelled",
                        ]:
                            break

                    time.sleep(1)

        # Final status
        final_job = self.pipeline.get_job_status(job_id)
        if final_job:
            self.display_job_summary(final_job)

        return final_job

    def display_job_summary(self, job: BatchJob):
        """Display comprehensive job summary"""
        console.print(f"\n📊 [bold blue]Job Summary: {job.name}[/bold blue]")

        # Status panel
        status_color = {
            "completed": "green",
            "completed_with_errors": "yellow",
            "failed": "red",
            "cancelled": "orange",
        }.get(job.status, "white")

        status_text = Text(job.status.upper(), style=status_color)
        console.print(f"Status: {status_text}")
        console.print(
            f"Duration: {(datetime.now() - job.created_at).total_seconds():.1f}s"
        )
        console.print(f"Total files: {job.total_files}")
        console.print(f"Successful: {job.completed_files}")
        console.print(f"Failed: {job.failed_files}")

        if job.results:
            total_predictions = sum(len(r.get("predictions", [])) for r in job.results)
            console.print(f"Total predictions: {total_predictions}")

        # Show errors if any
        if job.error_messages:
            console.print(
                f"\n⚠️ [bold red]Errors ({len(job.error_messages)}):[/bold red]"
            )
            for i, error in enumerate(job.error_messages[-5:], 1):  # Show last 5 errors
                console.print(f"  {i}. {error}", style="red")

            if len(job.error_messages) > 5:
                console.print(f"  ... and {len(job.error_messages) - 5} more errors")

        # Output information
        if job.status in ["completed", "completed_with_errors"]:
            console.print(
                f"\n📁 [bold green]Results saved to:[/bold green] {job.output_dir}"
            )

            # Check for result files
            results_dir = os.path.join(job.output_dir, "results")
            if os.path.exists(results_dir):
                result_files = [
                    f for f in os.listdir(results_dir) if f.endswith(".json")
                ]
                console.print(f"   - {len(result_files)} result files generated")

    def list_jobs(self):
        """List all jobs with their status"""
        if not self.pipeline:
            console.print("❌ Pipeline not initialized", style="red")
            return

        jobs = self.pipeline.list_jobs()

        if not jobs:
            console.print("📋 No jobs found", style="yellow")
            return

        console.print(f"\n📋 [bold blue]Batch Jobs ({len(jobs)})[/bold blue]")

        table = Table()
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Name", style="white")
        table.add_column("Status", style="white")
        table.add_column("Progress", style="white")
        table.add_column("Files", style="white")
        table.add_column("Created", style="white")

        for job in sorted(jobs, key=lambda x: x.created_at, reverse=True):
            status_style = {
                "completed": "green",
                "completed_with_errors": "yellow",
                "failed": "red",
                "cancelled": "orange",
                "running": "blue",
                "pending": "white",
            }.get(job.status, "white")

            table.add_row(
                job.job_id[:8] + "...",
                job.name,
                Text(job.status.upper(), style=status_style),
                f"{job.progress:.1f}%",
                f"{job.completed_files}/{job.total_files}",
                job.created_at.strftime("%m-%d %H:%M"),
            )

        console.print(table)

    def cancel_job(self, job_id: str):
        """Cancel a running job"""
        if not self.pipeline:
            console.print("❌ Pipeline not initialized", style="red")
            return

        if self.pipeline.cancel_job(job_id):
            console.print(f"✅ Job {job_id} cancelled", style="yellow")
        else:
            console.print(f"❌ Failed to cancel job {job_id}", style="red")

    def monitor_job(self, job_id: str):
        """Monitor a specific job"""
        if not self.pipeline:
            console.print("❌ Pipeline not initialized", style="red")
            return

        job = self.pipeline.get_job_status(job_id)
        if not job:
            console.print(f"❌ Job {job_id} not found", style="red")
            return

        console.print(f"\n👁️ [bold blue]Monitoring Job: {job.name}[/bold blue]")
        console.print("Press Ctrl+C to stop monitoring\n")

        try:
            while True:
                current_job = self.pipeline.get_job_status(job_id)
                if current_job:
                    # Clear screen and show status
                    os.system("clear" if os.name == "posix" else "cls")

                    console.print(
                        f"🔄 [bold blue]Live Job Monitor - {current_job.name}[/bold blue]"
                    )
                    console.print(
                        f"Last updated: {datetime.now().strftime('%H:%M:%S')}\n"
                    )

                    # Status table
                    table = Table(title="Job Status")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="white")

                    table.add_row("Job ID", current_job.job_id)
                    table.add_row("Status", current_job.status.upper())
                    table.add_row("Progress", f"{current_job.progress:.1f}%")
                    table.add_row("Completed Files", str(current_job.completed_files))
                    table.add_row("Failed Files", str(current_job.failed_files))
                    table.add_row("Total Files", str(current_job.total_files))
                    table.add_row(
                        "Created At",
                        current_job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    console.print(table)

                    # Recent errors
                    if current_job.error_messages:
                        console.print(
                            f"\n⚠️ Recent Errors ({len(current_job.error_messages)}):"
                        )
                        for error in current_job.error_messages[-3:]:
                            console.print(f"  - {error}", style="red")

                    # Check if job finished
                    if current_job.status in [
                        "completed",
                        "completed_with_errors",
                        "failed",
                        "cancelled",
                    ]:
                        console.print(
                            f"\n✅ Job finished with status: {current_job.status}"
                        )
                        break

                time.sleep(2)

        except KeyboardInterrupt:
            console.print("\n👋 Monitoring stopped")


# CLI Commands using Click
@click.group()
@click.option("--chunk-size", default=1000, help="Chunk size for streaming processing")
@click.option("--max-memory", default=512, help="Maximum memory usage per worker (MB)")
@click.option("--timeout", default=300, help="Timeout per file (seconds)")
@click.option(
    "--validation-level",
    default="moderate",
    type=click.Choice(["strict", "moderate", "lenient"]),
)
@click.pass_context
def cli(ctx, chunk_size, max_memory, timeout, validation_level):
    """Batch Prediction CLI - Process CSV files in batches with ML predictions"""
    ctx.ensure_object(dict)

    # Create configuration
    config = BatchProcessingConfig(
        chunk_size=chunk_size,
        max_memory_mb=max_memory,
        timeout_seconds=timeout,
        validation_level=validation_level,
        save_intermediate=True,
    )

    # Initialize CLI handler
    cli_handler = BatchCLI()
    cli_handler.setup_pipeline(config)

    ctx.obj["cli"] = cli_handler


@cli.command()
@click.argument("input_path")
@click.option("--output-dir", "-o", default="./batch_output", help="Output directory")
@click.option("--job-name", "-n", help="Job name")
@click.option("--batch-size", "-b", default=10, help="Batch size")
@click.option("--max-workers", "-w", default=3, help="Maximum worker threads")
@click.option(
    "--recursive/--no-recursive", default=True, help="Search recursively for CSV files"
)
@click.pass_context
def run(ctx, input_path, output_dir, job_name, batch_size, max_workers, recursive):
    """Run batch prediction on CSV files"""
    cli_handler = ctx.obj["cli"]

    # Find CSV files
    csv_files = cli_handler.find_csv_files(input_path, recursive)

    if not csv_files:
        console.print(f"❌ No CSV files found in: {input_path}", style="red")
        return

    console.print(f"📁 Found {len(csv_files)} CSV files")

    # Generate job name if not provided
    if not job_name:
        job_name = f"batch_{os.path.basename(input_path)}_{int(time.time())}"

    try:
        # Create job
        job_id = cli_handler.pipeline.create_batch_job(
            name=job_name,
            input_files=csv_files,
            output_dir=output_dir,
            batch_size=batch_size,
            max_workers=max_workers,
        )

        # Run job with progress tracking
        cli_handler.run_job_with_progress(job_id)

    except Exception as e:
        console.print(f"❌ Error running batch job: {e}", style="red")


@cli.command()
@click.pass_context
def create(ctx):
    """Create a new batch job interactively"""
    cli_handler = ctx.obj["cli"]
    job_id = cli_handler.create_job_interactive()

    if job_id:
        if click.confirm("Run the job now?"):
            cli_handler.run_job_with_progress(job_id)


@cli.command()
@click.pass_context
def list(ctx):
    """List all batch jobs"""
    cli_handler = ctx.obj["cli"]
    cli_handler.list_jobs()


@cli.command()
@click.argument("job_id")
@click.pass_context
def status(ctx, job_id):
    """Show detailed status of a job"""
    cli_handler = ctx.obj["cli"]

    job = cli_handler.pipeline.get_job_status(job_id)
    if job:
        cli_handler.display_job_summary(job)
    else:
        console.print(f"❌ Job {job_id} not found", style="red")


@cli.command()
@click.argument("job_id")
@click.pass_context
def monitor(ctx, job_id):
    """Monitor a job with live updates"""
    cli_handler = ctx.obj["cli"]
    cli_handler.monitor_job(job_id)


@cli.command()
@click.argument("job_id")
@click.pass_context
def cancel(ctx, job_id):
    """Cancel a running job"""
    cli_handler = ctx.obj["cli"]
    cli_handler.cancel_job(job_id)


@cli.command()
@click.argument("job_id")
@click.pass_context
def resume(ctx, job_id):
    """Resume a failed or cancelled job"""
    cli_handler = ctx.obj["cli"]

    job = cli_handler.pipeline.get_job_status(job_id)
    if not job:
        console.print(f"❌ Job {job_id} not found", style="red")
        return

    if job.status not in ["failed", "cancelled"]:
        console.print(f"❌ Job is not in a resumable state: {job.status}", style="red")
        return

    if click.confirm(f"Resume job '{job.name}'?"):
        cli_handler.run_job_with_progress(job_id)


if __name__ == "__main__":
    cli()
