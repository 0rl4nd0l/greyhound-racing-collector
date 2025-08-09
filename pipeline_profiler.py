#!/usr/bin/env python3
"""
Pipeline Profiling and Bottleneck Analysis Module
================================================

Comprehensive profiling system for the Greyhound Analysis Predictor pipeline:
- cProfile/pprofile analysis for CPU hotspots
- SQL query profiling with timing analysis
- I/O and network bottleneck detection
- Memory usage tracking
- Sequence diagram generation

Author: AI Assistant
Date: July 31, 2025
"""

import contextlib
import cProfile
import json
import os
import pstats
import sqlite3
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pprofile
import psutil


@dataclass
class ProfileResult:
    """Data class for profiling results."""

    function_name: str
    total_time: float
    cpu_time: float
    io_time: float
    memory_peak: float
    call_count: int
    bottlenecks: List[str]
    timestamp: str
    result: Any = None  # Store the actual function result


@dataclass
class SQLQueryProfile:
    """Data class for SQL query profiling."""

    query: str
    execution_time: float
    rows_affected: int
    is_slow: bool
    table_name: str
    operation_type: str
    timestamp: str


class PipelineProfiler:
    """
    Disabled pipeline profiler - No-op implementation to prevent conflicts.
    """

    def __init__(self, output_dir: str = "./profiling_results"):
        # Disabled profiler - no initialization
        self.output_dir = Path(output_dir)
        self.profile_results = []
        self.sql_profiles = []
        self.sequence_data = []
        # No actual profiling setup to avoid conflicts

    def profile_function(self, func, *args, **kwargs) -> ProfileResult:
        """
        Disabled profiling function - just execute the function without profiling.
        """
        function_name = func.__name__
        
        # Just execute the function without any profiling
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = None
            raise e
        
        # Return a dummy profile result
        return ProfileResult(
            function_name=function_name,
            total_time=0.0,
            cpu_time=0.0,
            io_time=0.0,
            memory_peak=0.0,
            call_count=0,
            bottlenecks=[],
            timestamp=datetime.now().isoformat(),
            result=result
        )

    def profile_sql_queries(self, db_path: str) -> List[SQLQueryProfile]:
        """
        Disabled SQL profiling - returns empty list.
        """
        return []

    def track_sequence_step(
        self,
        step_name: str,
        component: str,
        start_time: float,
        end_time: float,
        step_type: str = "processing",
        data_size: int = 0,
        **metadata,
    ):
        """
        Disabled sequence tracking - does nothing.
        """
        pass

    def generate_sequence_diagram(self) -> str:
        """
        Generate a sequence diagram from tracked steps.
        """
        print("📈 Generating sequence diagram...")

        if not self.sequence_data:
            print("⚠️  No sequence data available")
            return ""

        # Sort by start time
        sorted_steps = sorted(self.sequence_data, key=lambda x: x["start_time"])

        # Create PlantUML sequence diagram
        diagram = ["@startuml", "title Greyhound Predictor Data Flow Pipeline", ""]

        # Define participants
        participants = set(step["component"] for step in sorted_steps)
        for participant in sorted(participants):
            diagram.append(f"participant {participant}")

        diagram.append("")

        # Add sequence steps
        prev_component = None
        for step in sorted_steps:
            component = step["component"]
            step_name = step["step_name"]
            duration = step["duration"]
            step_type = step["step_type"]

            # Add arrows between components
            if prev_component and prev_component != component:
                diagram.append(f"{prev_component} -> {component}: {step_name}")

            # Add note with timing and type
            type_emoji = {
                "processing": "⚙️",
                "io": "💾",
                "network": "🌐",
                "db": "🗄️",
            }.get(step_type, "📊")

            diagram.append(
                f"note over {component}: {type_emoji} {step_name}\\n{duration:.3f}s"
            )

            prev_component = component

        diagram.append("@enduml")

        diagram_content = "\n".join(diagram)

        # Save diagram
        diagram_path = self.output_dir / "sequence_diagram.puml"
        with open(diagram_path, "w") as f:
            f.write(diagram_content)

        print(f"✅ Sequence diagram saved: {diagram_path}")
        return diagram_content

    def generate_timing_table(self) -> str:
        """
        Generate a comprehensive timing table with hotspot analysis.
        """
        print("📊 Generating timing table...")

        # Combine all timing data
        all_timings = []

        # Add function profiles
        for profile in self.profile_results:
            all_timings.append(
                {
                    "Component": "Function",
                    "Name": profile.function_name,
                    "Duration (s)": f"{profile.total_time:.3f}",
                    "CPU Time (s)": f"{profile.cpu_time:.3f}",
                    "I/O Time (ms)": f"{profile.io_time:.1f}",
                    "Memory (MB)": f"{profile.memory_peak:.2f}",
                    "Type": (
                        "CPU" if profile.cpu_time > profile.total_time * 0.7 else "I/O"
                    ),
                    "Hotspot": (
                        "🔥"
                        if profile.total_time > 1.0
                        else "⚡" if profile.total_time > 0.1 else "✅"
                    ),
                }
            )

        # Add SQL profiles
        for sql_profile in self.sql_profiles:
            if sql_profile.is_slow:
                all_timings.append(
                    {
                        "Component": "SQL",
                        "Name": f"{sql_profile.operation_type} on {sql_profile.table_name}",
                        "Duration (s)": f"{sql_profile.execution_time/1000:.3f}",
                        "CPU Time (s)": "-",
                        "I/O Time (ms)": f"{sql_profile.execution_time:.1f}",
                        "Memory (MB)": "-",
                        "Type": "DB",
                        "Hotspot": "🐌" if sql_profile.execution_time > 500 else "⚠️",
                    }
                )

        # Add sequence steps
        for step in self.sequence_data:
            hotspot = (
                "🔥"
                if step["duration"] > 2.0
                else "⚡" if step["duration"] > 0.5 else "✅"
            )
            all_timings.append(
                {
                    "Component": step["component"],
                    "Name": step["step_name"],
                    "Duration (s)": f"{step['duration']:.3f}",
                    "CPU Time (s)": "-",
                    "I/O Time (ms)": "-",
                    "Memory (MB)": "-",
                    "Type": step["step_type"].upper(),
                    "Hotspot": hotspot,
                }
            )

        # Sort by duration
        all_timings.sort(key=lambda x: float(x["Duration (s)"]), reverse=True)

        # Create table
        table_lines = []
        table_lines.append(
            "| Component | Name | Duration (s) | CPU Time (s) | I/O Time (ms) | Memory (MB) | Type | Hotspot |"
        )
        table_lines.append(
            "|-----------|------|--------------|--------------|---------------|-------------|------|---------|"
        )

        for timing in all_timings:
            line = f"| {timing['Component']} | {timing['Name'][:30]} | {timing['Duration (s)']} | {timing['CPU Time (s)']} | {timing['I/O Time (ms)']} | {timing['Memory (MB)']} | {timing['Type']} | {timing['Hotspot']} |"
            table_lines.append(line)

        table_content = "\n".join(table_lines)

        # Save table
        table_path = self.output_dir / "timing_table.md"
        with open(table_path, "w") as f:
            f.write("# Pipeline Timing Analysis\n\n")
            f.write("## Hotspot Legend\n")
            f.write("- 🔥 Critical Hotspot (>1s or >500ms SQL)\n")
            f.write("- ⚡ Minor Hotspot (>0.1s or >100ms SQL)\n")
            f.write("- 🐌 Slow SQL Query\n")
            f.write("- ⚠️ SQL Warning\n")
            f.write("- ✅ Acceptable Performance\n\n")
            f.write(table_content)

        print(f"✅ Timing table saved: {table_path}")
        return table_content

    def generate_comprehensive_report(self) -> str:
        """
        Disabled report generation - returns empty string.
        """
        return ""

        report_lines = []
        report_lines.append("# Greyhound Predictor Pipeline Bottleneck Analysis")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## Executive Summary")
        total_functions = len(self.profile_results)
        total_sql_queries = len(self.sql_profiles)
        slow_sql_queries = len([p for p in self.sql_profiles if p.is_slow])
        total_sequence_steps = len(self.sequence_data)

        report_lines.append(f"- **Functions Profiled**: {total_functions}")
        report_lines.append(f"- **SQL Queries Analyzed**: {total_sql_queries}")
        report_lines.append(f"- **Slow SQL Queries**: {slow_sql_queries}")
        report_lines.append(f"- **Sequence Steps Tracked**: {total_sequence_steps}")
        report_lines.append("")

        # Critical Bottlenecks
        report_lines.append("## Critical Bottlenecks")
        critical_bottlenecks = []

        for profile in self.profile_results:
            if profile.total_time > 1.0:
                critical_bottlenecks.append(
                    f"🔥 **{profile.function_name}**: {profile.total_time:.3f}s (CPU: {profile.cpu_time:.3f}s)"
                )

        for sql_profile in self.sql_profiles:
            if sql_profile.execution_time > 500:
                critical_bottlenecks.append(
                    f"🐌 **SQL {sql_profile.operation_type}**: {sql_profile.execution_time:.2f}ms on {sql_profile.table_name}"
                )

        if critical_bottlenecks:
            report_lines.extend(critical_bottlenecks)
        else:
            report_lines.append("✅ No critical bottlenecks detected")

        report_lines.append("")

        # Recommendations
        report_lines.append("## Optimization Recommendations")

        # CPU recommendations
        cpu_heavy = [
            p
            for p in self.profile_results
            if p.cpu_time > p.total_time * 0.7 and p.total_time > 0.1
        ]
        if cpu_heavy:
            report_lines.append("### CPU Optimization")
            for profile in cpu_heavy:
                report_lines.append(
                    f"- Optimize **{profile.function_name}** - CPU bound ({profile.cpu_time:.3f}s)"
                )

        # I/O recommendations
        io_heavy = [p for p in self.profile_results if p.io_time > 100]  # >100ms I/O
        if io_heavy:
            report_lines.append("### I/O Optimization")
            for profile in io_heavy:
                report_lines.append(
                    f"- Optimize **{profile.function_name}** - I/O bound ({profile.io_time:.1f}ms)"
                )

        # SQL recommendations
        if slow_sql_queries:
            report_lines.append("### Database Optimization")
            report_lines.append("- Add indexes to frequently queried columns")
            report_lines.append("- Consider query optimization for slow operations")
            report_lines.append("- Implement connection pooling")

        # Memory recommendations
        memory_heavy = [
            p for p in self.profile_results if p.memory_peak > 100
        ]  # >100MB
        if memory_heavy:
            report_lines.append("### Memory Optimization")
            for profile in memory_heavy:
                report_lines.append(
                    f"- Optimize **{profile.function_name}** - High memory usage ({profile.memory_peak:.2f}MB)"
                )

        report_lines.append("")

        # Detailed Data
        report_lines.append("## Detailed Profiling Data")
        report_lines.append("### Function Profiles")
        for profile in sorted(
            self.profile_results, key=lambda x: x.total_time, reverse=True
        ):
            report_lines.append(f"- **{profile.function_name}**:")
            report_lines.append(f"  - Total Time: {profile.total_time:.3f}s")
            report_lines.append(f"  - CPU Time: {profile.cpu_time:.3f}s")
            report_lines.append(f"  - I/O Time: {profile.io_time:.1f}ms")
            report_lines.append(f"  - Memory Peak: {profile.memory_peak:.2f}MB")
            report_lines.append(f"  - Function Calls: {profile.call_count}")
            if profile.bottlenecks:
                report_lines.append(
                    f"  - Bottlenecks: {', '.join(profile.bottlenecks)}"
                )
            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "bottleneck_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"✅ Comprehensive report saved: {report_path}")
        return report_content

    def _analyze_bottlenecks(
        self, profiler_output: str, total_time: float
    ) -> List[str]:
        """
        Analyze profiler output to identify bottlenecks.
        """
        bottlenecks = []
        lines = profiler_output.split("\n")

        for line in lines:
            if "cumulative" in line or "tottime" in line:
                continue

            # Look for high time functions
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    cumtime = float(parts[3])
                    if cumtime > total_time * 0.1:  # More than 10% of total time
                        func_name = parts[-1]
                        bottlenecks.append(f"{func_name}({cumtime:.3f}s)")
                except (ValueError, IndexError):
                    continue

        return bottlenecks[:5]  # Top 5 bottlenecks

    def _save_profile_result(self, profile_result: ProfileResult, profiler_output: str):
        """
        Save detailed profile result to file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profile_{profile_result.function_name}_{timestamp}.txt"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(f"Profile for {profile_result.function_name}\n")
            f.write(f"Timestamp: {profile_result.timestamp}\n")
            f.write(f"Total Time: {profile_result.total_time:.3f}s\n")
            f.write(f"CPU Time: {profile_result.cpu_time:.3f}s\n")
            f.write(f"I/O Time: {profile_result.io_time:.1f}ms\n")
            f.write(f"Memory Peak: {profile_result.memory_peak:.2f}MB\n")
            f.write(f"Function Calls: {profile_result.call_count}\n")
            f.write(f"Bottlenecks: {', '.join(profile_result.bottlenecks)}\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("Detailed cProfile Output:\n")
            f.write(profiler_output)

    def _save_sql_profiles(self, sql_profiles: List[SQLQueryProfile]):
        """
        Save SQL profiles to JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sql_profiles_{timestamp}.json"
        filepath = self.output_dir / filename

        profiles_data = [asdict(profile) for profile in sql_profiles]

        with open(filepath, "w") as f:
            json.dump(profiles_data, f, indent=2)


# Global profiler instance
pipeline_profiler = PipelineProfiler()


def profile_function(func):
    """
    Disabled profiling decorator - returns function unchanged.
    """
    return func  # Just return the function unchanged


def track_sequence(step_name: str, component: str, step_type: str = "processing"):
    """
    Disabled context manager - returns dummy context manager.
    """
    
    class DummySequenceTracker:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass  # Do nothing
    
    return DummySequenceTracker()


if __name__ == "__main__":
    print("🔍 Pipeline Profiler Module Loaded")
    print("Usage:")
    print(
        "  from pipeline_profiler import profile_function, track_sequence, pipeline_profiler"
    )
    print("  @profile_function")
    print("  def my_function(): ...")
    print("  with track_sequence('step_name', 'component', 'processing'): ...")
