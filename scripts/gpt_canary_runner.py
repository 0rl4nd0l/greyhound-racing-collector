#!/usr/bin/env python3
"""
Periodic GPT canary runner

Usage:
  python3 scripts/gpt_canary_runner.py [--interval 900] [--retention-days 7] [--max-files 500]

Reads OPENAI_API_KEY from the environment via app import (dotenv).
Writes JSON snapshots under logs/diagnostics/gpt/canary_YYYYmmdd_HHMMSS.json
After each run, archives older diagnostics into logs/diagnostics/gpt/archives to keep the folder tidy.
"""
from __future__ import annotations
import os
import sys
import json
import time
import shutil
from datetime import datetime
from typing import Tuple

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app  # noqa: E402


DIAG_BASE = os.path.join('logs', 'diagnostics', 'gpt')
ARCHIVES_DIR = os.path.join(DIAG_BASE, 'archives')


def _list_json_files(dir_path: str) -> list[Tuple[str, float]]:
    try:
        items = []
        for name in os.listdir(dir_path):
            if not name.endswith('.json'):
                continue
            fp = os.path.join(dir_path, name)
            try:
                st = os.stat(fp)
                items.append((fp, st.st_mtime))
            except Exception:
                continue
        items.sort(key=lambda x: x[1])  # oldest first
        return items
    except FileNotFoundError:
        return []


def archive_old_canary_files(base_dir: str = DIAG_BASE, retention_days: int = 7, max_files: int = 500) -> dict:
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    now = time.time()
    threshold = now - max(1, retention_days) * 24 * 3600
    files = _list_json_files(base_dir)
    archived = []
    # Identify files older than threshold
    old_files = [fp for fp, m in files if m < threshold]
    # If too many files, also archive the oldest beyond the most recent max_files
    overflow = max(0, len(files) - max_files)
    overflow_files = [fp for fp, _ in files[:overflow]] if overflow else []
    targets = []
    seen = set()
    for fp in old_files + overflow_files:
        if fp not in seen:
            seen.add(fp)
            targets.append(fp)
    for fp in targets:
        base = os.path.basename(fp)
        name_no_ext = os.path.splitext(base)[0]
        zip_base = os.path.join(ARCHIVES_DIR, name_no_ext)
        try:
            # Create a zip containing only this file
            # shutil.make_archive requires a directory; we place a temp copy
            tmp_dir = os.path.join(ARCHIVES_DIR, f".tmp_{name_no_ext}")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_fp = os.path.join(tmp_dir, base)
            try:
                shutil.copy2(fp, tmp_fp)
                shutil.make_archive(zip_base, 'zip', tmp_dir)
                os.remove(fp)
                archived.append(base)
            finally:
                # cleanup temp
                try:
                    os.remove(tmp_fp)
                except Exception:
                    pass
                try:
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
        except Exception:
            # Non-fatal
            continue
    return {"archived_files": archived, "archives_dir": ARCHIVES_DIR}


def run_canary_once() -> dict:
    client = app.test_client()
    resp = client.get('/api/gpt/test')
    try:
        data = resp.get_json() or {}
    except Exception:
        data = {'success': False, 'error': 'parse_error', 'status_code': resp.status_code}
    # Persist to diagnostics directory
    out_dir = DIAG_BASE
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'canary_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Archive old files to keep directory small
    archive_info = archive_old_canary_files(base_dir=out_dir)
    return {
        'path': out_path,
        'success': bool(data.get('success', True)),
        'status': data,
        'archive': archive_info,
    }


def main(argv: list[str]) -> int:
    # Simple argv parser for --interval seconds
    interval = 900
    retention_days = 7
    max_files = 500
    i = 0
    while i < len(argv):
        if argv[i] == '--interval' and i + 1 < len(argv):
            try:
                interval = int(argv[i+1])
            except Exception:
                pass
            i += 2
        elif argv[i] == '--retention-days' and i + 1 < len(argv):
            try:
                retention_days = int(argv[i+1])
            except Exception:
                pass
            i += 2
        elif argv[i] == '--max-files' and i + 1 < len(argv):
            try:
                max_files = int(argv[i+1])
            except Exception:
                pass
            i += 2
        else:
            i += 1
    print(f"[canary] Starting periodic GPT canary; interval={interval}s, retention_days={retention_days}, max_files={max_files}")
    try:
        while True:
            result = run_canary_once()
            # Re-archive with configured parameters (in case defaults differ)
            try:
                archive_info = archive_old_canary_files(base_dir=DIAG_BASE, retention_days=retention_days, max_files=max_files)
                if archive_info.get('archived_files'):
                    print(f"[canary] Archived {len(archive_info['archived_files'])} old files -> {archive_info['archives_dir']}")
            except Exception as e:
                print(f"[canary] Archive step failed: {e}")
            print(f"[canary] Wrote {result['path']} (success={result['success']})")
            time.sleep(max(5, interval))
    except KeyboardInterrupt:
        print("[canary] Stopped by user")
        return 0
    except Exception as e:
        print(f"[canary] Error: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

