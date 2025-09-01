#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
import traceback

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_system_v4 import MLSystemV4 as V4


def find_db_candidates():
    # Prefer analytics DB for reads
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V4 sanity check: prepare_time_ordered_data on a small slice')
    parser.add_argument('--max-races', type=int, default=200, help='Limit number of races for quick check')
    args = parser.parse_args()

    results = {
        'project_root': str(PROJECT_ROOT),
        'env_db': os.getenv('GREYHOUND_DB_PATH'),
        'db_candidates': [],
        'used_db': None,
        'ok': False,
        'message': '',
        'summary': {}
    }

    # Discover DB
    cands = find_db_candidates()
    results['db_candidates'] = cands
    if not cands:
        results['message'] = 'No database candidates found. Set GREYHOUND_DB_PATH.'
        print(json.dumps(results, indent=2))
        return 1

    db = cands[0]
    os.environ['GREYHOUND_DB_PATH'] = db
    results['used_db'] = db
    os.environ['V4_MAX_RACES'] = str(max(1, args.max_races))

    try:
        sys_v4 = V4()
        train_df, test_df = sys_v4.prepare_time_ordered_data()

        train_races = len(train_df['race_id'].unique()) if not train_df.empty and 'race_id' in train_df.columns else 0
        test_races = len(test_df['race_id'].unique()) if not test_df.empty and 'race_id' in test_df.columns else 0

        results['summary'] = {
            'train_samples': int(len(train_df)),
            'test_samples': int(len(test_df)),
            'train_races': int(train_races),
            'test_races': int(test_races),
        }

        if train_df.empty or test_df.empty:
            results['ok'] = False
            results['message'] = 'prepare_time_ordered_data returned empty splits (check data presence and filters)'
            print(json.dumps(results, indent=2))
            return 1

        results['ok'] = True
        results['message'] = 'Sanity check succeeded'
        print(json.dumps(results, indent=2))
        return 0
    except Exception as e:
        results['ok'] = False
        results['message'] = f'Exception: {e}'
        results['trace'] = traceback.format_exc()
        print(json.dumps(results, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

