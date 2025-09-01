#!/usr/bin/env python3
"""
Chromium Process Reaper
======================

Manual cleanup script for orphaned Chrome, Chromium, and ChromeDriver processes.
Useful for dev/CI environments and emergency cleanup.

Usage:
    python scripts/ops/reap_chromium.py [--dry-run] [--grace-seconds=N]
"""

import argparse
import psutil
import time
import sys


def reap_chromium_processes(dry_run=False, grace_seconds=0):
    """
    Kill orphaned Chromium-related processes.
    
    Args:
        dry_run: If True, just print what would be killed
        grace_seconds: Only kill processes older than this
        
    Returns:
        Number of processes killed/identified
    """
    killed = 0
    now = time.time()
    
    print("🔍 Scanning for orphaned Chromium processes...")
    
    for p in psutil.process_iter(["name", "ppid", "create_time", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            ppid = p.info.get("ppid", 0)
            create_time = p.info.get("create_time", now)
            cmdline = " ".join(p.info.get("cmdline", []))
            
            # Match Chrome/Chromium/ChromeDriver processes
            if not any(k in name for k in ("chromedriver", "chrome", "chromium")):
                continue
            
            # Check if orphaned (parent is init or doesn't exist)
            is_orphaned = ppid in (0, 1)
            
            # Check age if grace period specified
            age_seconds = now - create_time
            is_old_enough = grace_seconds == 0 or age_seconds > grace_seconds
            
            if is_orphaned and is_old_enough:
                if dry_run:
                    print(f"  [DRY RUN] Would kill PID {p.pid}: {name} (parent: {ppid}, age: {age_seconds:.1f}s)")
                    print(f"            Command: {cmdline[:100]}...")
                else:
                    try:
                        p.kill()
                        print(f"  ☠️  Killed PID {p.pid}: {name} (parent: {ppid}, age: {age_seconds:.1f}s)")
                        killed += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"  ⚠️  Failed to kill PID {p.pid}: {e}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process disappeared or access denied, skip
            continue
        except Exception as e:
            print(f"  ⚠️  Error processing PID {getattr(p, 'pid', '?')}: {e}")
            continue
    
    return killed


def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned Chromium processes")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be killed without actually killing")
    parser.add_argument("--grace-seconds", type=int, default=0,
                       help="Only kill processes older than this many seconds")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show additional information")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Grace period: {args.grace_seconds} seconds")
        print(f"Dry run: {args.dry_run}")
        print()
    
    killed = reap_chromium_processes(
        dry_run=args.dry_run,
        grace_seconds=args.grace_seconds
    )
    
    if args.dry_run:
        print(f"\n✅ Found {killed} orphaned Chromium processes (dry run)")
    else:
        print(f"\n✅ Killed {killed} orphaned Chromium processes")
    
    return 0 if killed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
