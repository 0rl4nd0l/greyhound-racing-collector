#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import tempfile
import traceback
from pathlib import Path

# Ensure local project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_system_v4 import MLSystemV4 as V4

REQUIRED_TABLES = ('dog_race_data', 'race_metadata', 'enhanced_expert_data')


def find_db_candidates():
    # Heuristics similar to app usage, prefer analytics DB for reads
    candidates = [
        os.getenv('ANALYTICS_DB_PATH'),
        os.getenv('GREYHOUND_DB_PATH'),
        'greyhound_racing_data.db',
        str(Path('databases') / 'comprehensive_greyhound_data.db'),
        str(Path('databases') / 'greyhound_racing_data.db'),
    ]
    out = []
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            out.append(str(p))
    return out


def run_preflight_method_check(db_path: str, expect_raise: bool, expect_missing: bool):
    # Bypass __init__ to directly exercise preflight on arbitrary db_path
    obj = object.__new__(V4)
    # attach only the needed attribute
    obj.db_path = db_path
    try:
        res = V4._preflight_check_required_tables(obj, required=REQUIRED_TABLES, raise_on_fail=expect_raise)
        if expect_raise:
            return False, 'Expected raise but got return', res
        # No raise
        if expect_missing:
            if not res or 'missing' not in res:
                return False, f"Expected missing tables but got {res}", res
            return True, 'Missing tables detected as expected', res
        else:
            if res is None:
                return True, 'Preflight OK', None
            return False, f"Unexpected missing result: {res}", res
    except Exception as e:
        if expect_raise:
            return True, f'Raised as expected: {e}', None
        return False, f'Unexpected exception: {e}', None


def main():
    results = {
        'project_root': str(PROJECT_ROOT),
        'env_db': os.getenv('GREYHOUND_DB_PATH'),
        'db_candidates': [],
        'tests': {}
    }

    # Collect candidates
    cands = find_db_candidates()
    results['db_candidates'] = cands

    # 1) Constructor pass test (only if we have a DB candidate)
    constructor_test = {'skipped': False, 'ok': False, 'message': '', 'db': None}
    if cands:
        db = cands[0]
        constructor_test['db'] = db
        try:
            # Ensure env var is set for this construction path (V4 still uses GREYHOUND_DB_PATH)
            os.environ['GREYHOUND_DB_PATH'] = db
            _ = V4()  # will run constructor preflight
            constructor_test['ok'] = True
            constructor_test['message'] = 'Constructor preflight passed with detected DB'
        except Exception as e:
            constructor_test['ok'] = False
            constructor_test['message'] = f'Constructor failed: {e}'
    else:
        constructor_test['skipped'] = True
        constructor_test['message'] = 'No DB candidates found. Set GREYHOUND_DB_PATH to a valid DB and re-run.'
    results['tests']['constructor_pass'] = constructor_test

    # 2) Missing DB should raise
    missing_path = str(PROJECT_ROOT / 'tmp_nonexistent__ghdash__preflight.db')
    if Path(missing_path).exists():
        try:
            Path(missing_path).unlink()
        except Exception:
            pass
    ok, msg, payload = run_preflight_method_check(missing_path, expect_raise=True, expect_missing=False)
    results['tests']['missing_db_raises'] = {'ok': ok, 'message': msg, 'path': missing_path}

    # 3) Empty DB exists but should report missing tables
    with tempfile.TemporaryDirectory() as td:
        empty_db = str(Path(td) / 'empty.db')
        sqlite3.connect(empty_db).close()
        ok2, msg2, payload2 = run_preflight_method_check(empty_db, expect_raise=False, expect_missing=True)
        results['tests']['empty_db_missing_tables'] = {'ok': ok2, 'message': msg2, 'path': empty_db, 'missing': (payload2 or {}).get('missing')}

    # Overall
    overall_ok = True
    for k, t in results['tests'].items():
        if isinstance(t, dict) and not t.get('skipped', False):
            if not t.get('ok', False):
                overall_ok = False
                break

    print(json.dumps({'overall_ok': overall_ok, **results}, indent=2))
    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())

