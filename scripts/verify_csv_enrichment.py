#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from app import app  # noqa: E402

RACE_FILENAME = os.environ.get('RACE_FILENAME')
if not RACE_FILENAME:
    # Fallback to first CSV in upcoming_races
    upcoming_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'upcoming_races'))
    candidates = [f for f in os.listdir(upcoming_dir) if f.endswith('.csv')]
    if not candidates:
        print('ERROR: No CSVs found in upcoming_races')
        sys.exit(2)
    RACE_FILENAME = candidates[0]

print(f'Using race file: {RACE_FILENAME}')

client = app.test_client()

payload = {"race_filename": RACE_FILENAME}
start = time.time()
resp = client.post('/api/predict_single_race_enhanced', json=payload)
elapsed = time.time() - start
print(f'HTTP {resp.status_code}, elapsed {elapsed:.2f}s')

try:
    data = resp.get_json(silent=True) if hasattr(resp, 'get_json') else None
except Exception:
    data = None

if not data:
    print('ERROR: No JSON in response')
    print(resp.data[:500])
    sys.exit(3)

print(f"success={data.get('success')} predictor_used={data.get('predictor_used')}")
# Collect predictions list from common keys
preds = None
container = data
# Some endpoints wrap content under 'prediction'
if isinstance(data.get('prediction'), dict):
    container = data['prediction']
for key in ('predictions', 'enhanced_predictions'):
    if isinstance(container.get(key), list):
        preds = container[key]
        break

if not preds:
    print('ERROR: No predictions list found in response keys')
    print('Keys:', list(data.keys()))
    print('Nested prediction keys:', list(container.keys()))
    sys.exit(4)

print(f'Predictions count: {len(preds)}')

# Check csv_* enrichment presence across predictions
csv_keys = [
    'csv_historical_races',
    'csv_win_rate',
    'csv_place_rate',
    'csv_avg_finish_position',
    'csv_best_finish_position',
    'csv_recent_form',
    'csv_avg_time',
    'csv_best_time',
]

present_counts = {k: 0 for k in csv_keys}
for p in preds:
    for k in csv_keys:
        if k in p and p[k] is not None:
            present_counts[k] += 1

print('API response enrichment key presence:')
for k in csv_keys:
    print(f'  {k}: {present_counts[k]}/{len(preds)}')

# Now verify persisted JSON contains same csv_* fields
pred_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'predictions'))
latest_path = None
latest_mtime = 0
if os.path.isdir(pred_dir):
    for name in os.listdir(pred_dir):
        if not name.endswith('.json'):
            continue
        p = os.path.join(pred_dir, name)
        try:
            mtime = os.path.getmtime(p)
        except Exception:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = p

if not latest_path:
    print('WARNING: No prediction JSON files found in ./predictions')
    sys.exit(0)

print(f'Latest prediction file: {os.path.basename(latest_path)} (mtime={datetime.fromtimestamp(latest_mtime).isoformat()})')

try:
    with open(latest_path, 'r', encoding='utf-8') as f:
        saved = json.load(f)
except Exception as e:
    print(f'ERROR: Failed to read saved prediction JSON: {e}')
    sys.exit(5)

saved_preds = None
for key in ('predictions', 'enhanced_predictions'):
    if isinstance(saved.get(key), list):
        saved_preds = saved[key]
        break

if not saved_preds:
    print('WARNING: No predictions list in saved file')
    sys.exit(0)

saved_counts = {k: 0 for k in csv_keys}
for p in saved_preds:
    for k in csv_keys:
        if k in p and p[k] is not None:
            saved_counts[k] += 1

print('Saved file enrichment key presence:')
for k in csv_keys:
    print(f'  {k}: {saved_counts[k]}/{len(saved_preds)}')

print('Done.')

