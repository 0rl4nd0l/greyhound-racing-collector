"""Default-off development derivation of one production parent's two strengths.

The caller authenticates the common binding against retained native seals before
calling this pure boundary. Hash references bind provenance; they do not by
themselves prove that external history bytes were qualified. No files are read
or written here, and no new pre-jump or scientific membership is created.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import re

from src.predictor.market_form_residual import score_race

PARENT_FAMILY = 'market_form_residual_v1'
PARENT_MODEL_SHA256 = '624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d'
PARENT_MANIFEST_SHA256 = '8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080'
SCORER_SOURCE_SHA256 = '50039cbc46f48d2f2d0dda8973e75dc73055872ff74b082821535060cc36b7f6'
PLAN_FIELDS = frozenset(('schema_version', 'parent_family', 'model_sha256', 'manifest_sha256',
                        'effective_state_sha256', 'race_id', 'common_binding_sha256', 'strengths',
                        'runtime_sha256', 'scorer_source_sha256'))
BINDING_FIELDS = frozenset(('schema_version', 'race_id', 'original_record_sha256',
    'feature_source_sha256', 'odds_source_sha256', 'history_snapshot_sha256',
    'retained_manifest_sha256', 'original_sealed_at', 'original_source_commit'))


def _canonical(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))+'\n').encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _time(value):
    dt = datetime.fromisoformat(value)
    if dt.utcoffset() is None:
        raise ValueError('controlled_timestamp_timezone_required')
    return dt


def derive_controlled_pair(frozen, original_record, binding, plan, *, derived_at, execute=False):
    """Replay a hash-bound original shadow record; emit a new development pair.

    ``execute`` is default OFF, not a grant of permission. Protected cohort
    membership and retained-input verification remain the root caller's duty.
    ``derived_at`` is the actual invocation timestamp; the native score time is
    retained only as historical replay provenance, never passed off as now.
    """
    if execute is not True:
        raise ValueError('controlled_derivation_disabled')
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS or plan['schema_version'] != 'controlled_adjustment_plan_v1':
        raise ValueError('controlled_plan_contract')
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS or binding['schema_version'] != 'controlled_adjustment_common_binding_v1':
        raise ValueError('controlled_binding_contract')
    for key, value in [*plan.items(), *binding.items()]:
        if key.endswith('_sha256') and (not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value)):
            raise ValueError('controlled_hash_invalid')
    if (plan['parent_family'] != PARENT_FAMILY or plan['model_sha256'] != PARENT_MODEL_SHA256
            or plan['manifest_sha256'] != PARENT_MANIFEST_SHA256 or plan['scorer_source_sha256'] != SCORER_SOURCE_SHA256 or plan['strengths'] != [1.0, 0.5]
            or any(type(v) not in (int, float) for v in plan['strengths'])):
        raise ValueError('controlled_parent_or_strength_changed')
    if any(getattr(frozen, key) != plan[key] for key in ('model_sha256', 'manifest_sha256', 'effective_state_sha256')):
        raise ValueError('controlled_parent_pin_mismatch')
    if _digest(binding) != plan['common_binding_sha256'] or _digest(original_record) != binding['original_record_sha256']:
        raise ValueError('controlled_common_binding_mismatch')
    if not re.fullmatch('[a-f0-9]{40}', binding['original_source_commit']):
        raise ValueError('controlled_original_source_pin_invalid')
    if not isinstance(derived_at, datetime) or derived_at.utcoffset() is None:
        raise ValueError('controlled_timestamp_timezone_required')
    if plan['race_id'] != binding['race_id'] or original_record.get('race_id') != plan['race_id']:
        raise ValueError('controlled_race_mismatch')
    sealed = _time(binding['original_sealed_at'])
    jump = _time(original_record['jump_timestamp'])
    if not _time(original_record['score_timestamp']) <= sealed <= jump - timedelta(seconds=120) or derived_at < sealed:
        raise ValueError('controlled_seal_timing_invalid')
    inputs = original_record['inputs']
    quotes = {_time(row['odds_capture_timestamp']) for row in inputs['runners']}
    if len(quotes) != 1 or not all(120 <= (jump - quote).total_seconds() <= 600 for quote in quotes):
        raise ValueError('controlled_quote_window_invalid')
    for row in inputs['runners']:
        if any(row[key] != binding[key] for key in ('feature_source_sha256', 'odds_source_sha256')):
            raise ValueError('controlled_source_inputs_mismatch')
    # Reuse the exact production implementation, including its artifact-state,
    # complete field, missingness, temporal, outcome and numerical checks.
    replay = score_race(frozen, inputs['runners'], inputs['provenance'])
    if _canonical(replay) != _canonical(original_record):
        raise ValueError('controlled_original_record_replay_mismatch')
    common = {'original_record_sha256': binding['original_record_sha256'],
              'inputs_sha256': _digest(inputs), 'binding_sha256': _digest(binding),
              'parent_effective_state_sha256': frozen.effective_state_sha256}
    common_sha = _digest(common)
    rows = replay['predictions']
    candidates = [{'candidate_id': f'production_parent_alpha_{label}', 'strength': strength,
                   'common_input_sha256': common_sha, 'parent_effective_state_sha256': frozen.effective_state_sha256,
                   'probabilities': [r[f'{kind}_probability'] for r in rows]}
                  for label, strength, kind in [('1_0', 1.0, 'full'), ('0_5', 0.5, 'half')]]
    output = {'schema_version': 'controlled_adjustment_pair_v1',
        'status': 'DEVELOPMENT_DERIVATION_NOT_AN_ORIGINAL_PREJUMP_FORECAST',
        'derived_at': derived_at.isoformat(), 'original_score_timestamp': original_record['score_timestamp'],
        'original_sealed_at': binding['original_sealed_at'], 'race_id': plan['race_id'],
        'parent_family': PARENT_FAMILY, 'parent_model_sha256': PARENT_MODEL_SHA256,
        'parent_manifest_sha256': PARENT_MANIFEST_SHA256, 'scorer_source_sha256': SCORER_SOURCE_SHA256,
        'runtime_sha256': plan['runtime_sha256'],
        'runtime_verification': 'REQUIRED_AT_CALLER_BOUNDARY',
        'plan_sha256': _digest(plan), 'common_binding': dict(binding), 'common_inputs': common,
        'runner_ids': [r['runner_id'] for r in rows], 'market_probabilities': [r['market_probability'] for r in rows],
        'candidates': candidates, 'only_varied_parameter': 'strength', 'fitting_performed': False,
        'history_reconstruction_performed': False, 'outcomes_present': False, 'performance_evaluated': False,
        'scientific_membership_created': False, 'original_forecasts_modified': False,
        'activation': False, 'persistence_performed': False,
        'external_history_qualification': 'REQUIRED_AT_CALLER_BOUNDARY_NOT_REPROVEN_BY_HASH_REFERENCE'}
    return {**output, 'derivation_sha256': _digest(output)}
