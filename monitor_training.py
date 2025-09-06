#!/usr/bin/env python3
"""
Model Training Progress Monitor
Checks training progress every 30 seconds
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime


def get_training_processes():
    """Get current training processes"""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = result.stdout.split("\n")

        training_procs = []
        for line in lines:
            if (
                any(
                    keyword in line.lower()
                    for keyword in ["train", "ml_system", "comprehensive_enhanced"]
                )
                and "grep" not in line
                and line.strip()
            ):
                # Extract PID, CPU, memory, and command
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    command = (
                        " ".join(parts[10:])[:100] + "..."
                        if len(" ".join(parts[10:])) > 100
                        else " ".join(parts[10:])
                    )
                    training_procs.append(
                        {"pid": pid, "cpu": cpu, "mem": mem, "command": command}
                    )

        return training_procs
    except Exception as e:
        print(f"Error getting processes: {e}")
        return []


def check_log_files():
    """Check recent log entries"""
    log_files = [
        "ml_pipeline_validation.log",
        "flask_output.log",
        "training.log",
        "ml_system.log",
    ]

    recent_logs = []
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                # Get last 5 lines
                result = subprocess.run(
                    ["tail", "-5", log_file], capture_output=True, text=True
                )
                if result.stdout.strip():
                    recent_logs.append(f"\n📄 {log_file}:")
                    recent_logs.append(result.stdout.strip())
            except Exception as e:
                pass

    return recent_logs


def monitor_training():
    """Main monitoring loop"""
    print("🚀 Starting Model Training Monitor")
    print("=" * 60)
    print("Checking every 30 seconds. Press Ctrl+C to stop.")
    print("=" * 60)

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n🕐 [{timestamp}] Update #{iteration}")
            print("-" * 50)

            # Check training processes
            processes = get_training_processes()
            if processes:
                print(f"🔥 Active Training Processes: {len(processes)}")
                for i, proc in enumerate(processes, 1):
                    print(
                        f"  {i}. PID: {proc['pid']} | CPU: {proc['cpu']}% | Memory: {proc['mem']}%"
                    )
                    print(f"     Command: {proc['command']}")
            else:
                print("⏸️  No active training processes found")

            # Check recent logs
            logs = check_log_files()
            if logs:
                print("\n📋 Recent Log Activity:")
                for log in logs[:10]:  # Limit output
                    print(log)

            # Check system resources
            try:
                load_avg = subprocess.run(
                    ["uptime"], capture_output=True, text=True
                ).stdout.strip()
                if "load average" in load_avg:
                    load_part = load_avg.split("load average:")[1].strip()
                    print(f"\n💻 System Load: {load_part}")
            except:
                pass

            print("-" * 50)
            print("⏳ Waiting 30 seconds for next update...")

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print("Training processes may still be running in background")


if __name__ == "__main__":
    monitor_training()
