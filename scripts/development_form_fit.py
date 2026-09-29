"""One fixed development fit with complete receipts; never called by production.

Callers must resolve permitted identities before loading rows. This API records
that admission evidence; it does not grant access to any additional population.
"""
from __future__ import annotations

from datetime import date
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np

from scripts.development_form_quality import CONTRACT, NAMES
from scripts import offline_systematic_search as framework


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def save(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return digest(payload)


def fit_diagnostic(rows, *, out: Path, training_end, evaluation_start, admission):
    """Fixed L2=1, 16 inputs, existing chronological residual fit; no search.

Retain failures and allocate a fresh output directory for every attempt. Saved
training predictions are replay checks, never evaluation evidence.
"""
    out.mkdir(parents=True, exist_ok=False)
    ledger = framework.Ledger(out / 'attempt.jsonl')
    ledger.append('START', variant='retained_card_form_v2_l2_1', development_fit=True)
    try:
        if not admission:
            raise ValueError('admission evidence required')
        if date.fromisoformat(training_end) >= date.fromisoformat(evaluation_start):
            raise ValueError('training must precede evaluation')
        rows = sorted(rows, key=lambda r: (r['race_date'], r['race_id'], r['box']))
        if not rows or any(date.fromisoformat(r['race_date']) > date.fromisoformat(training_end) for r in rows):
            raise ValueError('training membership exceeds cutoff')
        membership = [{k: r[k] for k in ('race_id', 'race_date', 'box', 'dog_token')} for r in rows]
        if len({(r['race_id'], r['box']) for r in rows}) != len(rows):
            raise ValueError('duplicate training member')
        if any(set(r['features']) != set(NAMES.values()) for r in rows):
            raise ValueError('feature contract mismatch')
        framework.validate(rows, np.asarray([r['market'] for r in rows], float))
        input_hash = save(out / 'training_rows.json', rows)
        admission_hash = save(out / 'admission.json', admission)
        contract_hash = save(out / 'feature_contract.json', CONTRACT)
        model = framework.linear_fit(rows, list(NAMES.values()), 1.0)
        predictions = framework.predict(rows, model)
        framework.validate(rows, predictions)
        code_files = [Path(__file__), Path(framework.__file__),
                      Path(__file__).with_name('development_form_quality.py'),
                      Path(__file__).with_name('offline_form_packet.py'),
                      Path(__file__).with_name('build_form_only_v1_packet.py')]
        receipt = {
            'identity': 'new_diagnostic_fit_not_original_artifact',
            'model': framework.serialize_model(model), 'training_membership': membership,
            'training_end': training_end, 'evaluation_start': evaluation_start,
            'input_sha256': input_hash, 'admission_sha256': admission_hash,
            'feature_contract_sha256': contract_hash, 'feature_contract': CONTRACT,
            'environment': {'python': sys.version, 'platform': platform.platform(),
                            'executable': sys.executable,
                            'executable_sha256': digest(Path(sys.executable).read_bytes()),
                            'packages': {name: importlib.metadata.version(name)
                                         for name in ('numpy', 'scipy', 'scikit-learn')},
                            'thread_settings': {k: os.environ.get(k) for k in
                                                ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')}},
            'code_sha256': {str(p): digest(p.read_bytes()) for p in code_files},
        }
        receipt_hash = save(out / 'receipt.json', receipt)
        save(out / 'training_predictions.json', predictions.tolist())
        ledger.append('COMPLETE', receipt_sha256=receipt_hash,
                      training_races=len({r['race_id'] for r in rows}), training_runners=len(rows))
        return predictions
    except Exception as exc:
        ledger.append('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
