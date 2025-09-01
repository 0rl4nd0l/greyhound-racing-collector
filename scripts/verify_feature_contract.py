#!/usr/bin/env python3
"""
verify_feature_contract.py
==========================

CI guard and local utility to validate the V4 feature contract against the current model.

Supports two modes:
- python: Use MLSystemV4 directly (no server required)
- api: Use a running Flask server's endpoints

Exit codes:
- 0: validation OK (or non-strict diff found)
- 1: strict mismatch, or fatal error in strict mode

Examples:
- Python mode (no server):
    python3 scripts/verify_feature_contract.py --refresh --strict

- API mode (server on http://localhost:5000):
    python3 scripts/verify_feature_contract.py --mode api --url http://localhost:5000 --strict

- JSON output (for CI parsers):
    python3 scripts/verify_feature_contract.py --json --strict
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure repository root is on sys.path so we can import ml_system_v4 when running from scripts/
try:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _compute_diff(cur_sig: str, cur_cats: set, cur_nums: set, exp: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    exp_sig = exp.get('feature_signature')
    exp_cats = set(exp.get('categorical_columns') or [])
    exp_nums = set(exp.get('numerical_columns') or [])

    signature_match = (exp_sig == cur_sig) if (exp_sig and cur_sig) else True
    cats_missing = sorted(list(exp_cats - cur_cats)) if exp_cats else []
    cats_extra = sorted(list(cur_cats - exp_cats)) if exp_cats else []
    nums_missing = sorted(list(exp_nums - cur_nums)) if exp_nums else []
    nums_extra = sorted(list(cur_nums - exp_nums)) if exp_nums else []

    matched = bool(signature_match and not cats_missing and not cats_extra and not nums_missing and not nums_extra)
    diff = {
        'signature_match': signature_match,
        'expected_signature': exp_sig,
        'current_signature': cur_sig,
        'categorical': {
            'missing': cats_missing,
            'extra': cats_extra,
        },
        'numerical': {
            'missing': nums_missing,
            'extra': nums_extra,
        },
    }
    return matched, diff


def check_via_python(strict: bool = False, refresh: bool = False) -> Dict[str, Any]:
    from importlib import import_module
    ML = import_module('ml_system_v4').MLSystemV4
    system = ML()

    # Optional refresh (regenerate contract file)
    refreshed_path = None
    if refresh:
        try:
            res = system.regenerate_feature_contract()
            if res.get('success'):
                refreshed_path = res.get('path')
        except Exception as e:
            if strict:
                return {'success': False, 'error': f'refresh_failed: {e}'}

    # Load expected contract
    cpath = Path('docs') / 'model_contracts' / 'v4_feature_contract.json'
    if not cpath.exists():
        return {'success': False, 'error': f'contract not found at {cpath}'}
    exp = _load_json(cpath)

    # Current model view
    cur_sig = system._compute_feature_signature(getattr(system, 'feature_columns', []) or [])
    cur_cats = set(getattr(system, 'categorical_columns', []) or [])
    cur_nums = set(getattr(system, 'numerical_columns', []) or [])

    matched, diff = _compute_diff(cur_sig, cur_cats, cur_nums, exp)
    ok = matched or (not strict)

    return {
        'success': ok,
        'matched': matched,
        'strict': bool(strict),
        'refreshed_path': refreshed_path,
        'diff': diff,
        'path': str(cpath),
    }


def _api_get(url: str) -> Tuple[int, Dict[str, Any]]:
    try:
        try:
            import requests  # type: ignore
            r = requests.get(url, headers={'Accept': 'application/json'}, timeout=30)
            return r.status_code, r.json()
        except ImportError:
            from urllib.request import urlopen, Request
            import json as _json
            req = Request(url, headers={'Accept': 'application/json'})
            with urlopen(req, timeout=30) as resp:  # nosec - CI localhost usage only
                status = getattr(resp, 'status', 200)
                data = _json.loads(resp.read().decode('utf-8'))
                return status, data
    except Exception as e:
        return 0, {'success': False, 'error': str(e)}


def _api_post(url: str) -> Tuple[int, Dict[str, Any]]:
    try:
        try:
            import requests  # type: ignore
            r = requests.post(url, headers={'Accept': 'application/json'}, timeout=30)
            return r.status_code, r.json()
        except ImportError:
            from urllib.request import urlopen, Request
            import json as _json
            req = Request(url, method='POST', headers={'Accept': 'application/json'})
            with urlopen(req, timeout=30) as resp:  # nosec - CI localhost usage only
                status = getattr(resp, 'status', 200)
                data = _json.loads(resp.read().decode('utf-8'))
                return status, data
    except Exception as e:
        return 0, {'success': False, 'error': str(e)}


def check_via_api(base_url: str, strict: bool = False, refresh: bool = False) -> Dict[str, Any]:
    if refresh:
        status, body = _api_post(f'{base_url}/api/v4/models/contracts/refresh')
        if status != 200 or not body.get('success'):
            return {'success': False, 'error': f'refresh_failed: status={status}, body={body}'}

    q = '?strict=1' if strict else ''
    status, body = _api_get(f'{base_url}/api/v4/models/contracts/check{q}')
    if status == 0:
        return {'success': False, 'error': body.get('error') if isinstance(body, dict) else 'unknown'}

    # API returns matched + diff + path under success true/false and 200/409
    ok = (status == 200) or (status == 409 and not strict)
    return {
        'success': ok and bool(body.get('success', True)),
        'matched': body.get('matched'),
        'strict': bool(body.get('strict')),
        'diff': body.get('diff'),
        'path': body.get('path'),
        'status': status,
    }


def main():
    ap = argparse.ArgumentParser(description='Validate V4 feature contract against the current model')
    ap.add_argument('--mode', choices=['python', 'api'], default='python', help='Validation mode (python: no server; api: use running app)')
    ap.add_argument('--url', default='http://localhost:5000', help='Base URL for API mode')
    ap.add_argument('--refresh', action='store_true', help='Regenerate the contract before checking')
    ap.add_argument('--strict', action='store_true', help='Exit non-zero on mismatch (strict mode)')
    ap.add_argument('--json', action='store_true', help='Emit JSON to stdout')
    ap.add_argument('-v', '--verbose', action='store_true', help='Verbose logging to stdout')

    args = ap.parse_args()

    try:
        if args.mode == 'api':
            result = check_via_api(args.url.rstrip('/'), args.strict, args.refresh)
        else:
            result = check_via_python(args.strict, args.refresh)
    except Exception as e:
        result = {'success': False, 'error': str(e)}

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if not result.get('success') and result.get('error'):
            print(f"ERROR: {result['error']}")
        else:
            matched = result.get('matched')
            print(f"Contract validation: {'MATCHED' if matched else 'MISMATCH'} (strict={result.get('strict')})")
            if 'path' in result:
                print(f"Contract file: {result['path']}")
            diff = result.get('diff') or {}
            if diff:
                print("Signature match:", diff.get('signature_match'))
                cat = diff.get('categorical') or {}
                num = diff.get('numerical') or {}
                print("Categorical missing:", cat.get('missing'))
                print("Categorical extra:", cat.get('extra'))
                print("Numerical missing:", num.get('missing'))
                print("Numerical extra:", num.get('extra'))

    # Exit code
    if result.get('success'):
        sys.exit(0)
    # If strict, treat mismatch/error as failure; otherwise pass
    sys.exit(1 if args.strict else 0)


if __name__ == '__main__':
    main()

