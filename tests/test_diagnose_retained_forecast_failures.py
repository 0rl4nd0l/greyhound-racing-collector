"""Offline regression: the July validator must accept only proven grade aliases."""
import hashlib
import importlib.util
import json
from pathlib import Path

import unittest

from scripts.diagnose_retained_forecast_failures import (
    corrected_feature_packet,
    load_grade_aliases,
    validate_independently,
)


class Reject(ValueError):
    pass


class NativeSeam:
    """Minimal native checks; retained integration runs the complete original function."""
    ManualPredictionError = Reject

    @staticmethod
    def _feature_packet(**kwargs):
        rows = json.loads(kwargs['feature_rows_raw'])
        assert hashlib.sha256(kwargs['feature_rows_raw']).hexdigest() == kwargs['feature_rows_sha']
        for row in rows:
            if not row['metadata_is_leakage_safe']:
                raise Reject('feature_row_metadata_not_safe')
            if row['target_grade_safe'].replace(' ', '').upper() != kwargs['context']['target_grade'].replace(' ', '').upper():
                raise Reject('feature_row_target_grade_mismatch')
        return rows, None, None


def packet(grade='Grade 6', safe=True):
    raw = json.dumps([{'race_id':'test', 'target_grade_safe':grade, 'metadata_is_leakage_safe':safe}]).encode()
    return {'race_id':'test', 'context':{'target_grade':'6th Grade'}, 'feature_rows_raw':raw, 'feature_rows_sha':hashlib.sha256(raw).hexdigest()}


def aliases():
    return load_grade_aliases(Path(__file__).parents[1] / 'scripts/predict_market_form_residual.py')


class FailureRepairTests(unittest.TestCase):
    def test_native_rejects_then_adapter_accepts_without_mutation(self):
        kwargs = packet()
        with self.assertRaisesRegex(Reject, 'target_grade_mismatch'):
            NativeSeam._feature_packet(**kwargs)
        before = dict(kwargs['context'])
        corrected_feature_packet(NativeSeam, kwargs, aliases())
        self.assertEqual(kwargs['context'], before)

    def test_different_unknown_ambiguous_missing_grade_blocked(self):
        for grade in ['Grade 5', 'Made-up grade', 'M', '']:
            with self.subTest(grade=grade), self.assertRaisesRegex(Reject, 'target_grade_mismatch'):
                corrected_feature_packet(NativeSeam, packet(grade), aliases())

    def test_safety_gate_not_bypassed(self):
        with self.assertRaisesRegex(Reject, 'metadata_not_safe'):
            corrected_feature_packet(NativeSeam, packet(safe=False), aliases())

    def test_one_rejection_does_not_abort_or_erase_other_dispositions(self):
        calls = []
        def validate(item):
            calls.append(item)
            if item == 'bad':
                raise Reject('feature_row_metadata_not_safe')
            return item
        result = validate_independently(['bad', 'good'], validate, Reject)
        self.assertEqual(calls, ['bad', 'good'])
        self.assertEqual(result, [{'status':'BLOCKED', 'reason':'feature_row_metadata_not_safe'}, {'status':'VALIDATED', 'result':'good'}])

    def test_unexpected_bug_not_disguised_as_input_exclusion(self):
        def broken(_):
            raise TypeError('defect')
        with self.assertRaises(TypeError):
            validate_independently(['bad'], broken, Reject)

    def test_native_retained_full_validator_regression(self):
        # Optional retained integration fixture: no network, DB, models or outcomes.
        from scripts.diagnose_retained_forecast_failures import load_native
        manifest = Path('/mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/inventory/july-operational-coverage.json')
        boundary = manifest.with_name('access-boundary.json')
        if not manifest.exists() or not boundary.exists():
            self.skipTest('retained audit corpus unavailable')
        protected = set(json.loads(boundary.read_text())['canonical_protected_identity_keys'])
        inventory = json.loads(manifest.read_text())
        successes = {r['canonical_race_key'] for r in inventory['races'] if r['any_append']}
        attempt = next(a for a in inventory['attempts'] if not a['reserved'] and a['canonical_race_key'] not in protected and a['canonical_race_key'] not in successes and a['failure_reason'] == 'feature_row_target_grade_mismatch')
        stage = json.loads(Path(attempt['path']).read_text())
        original = next(r for r in stage['races'] if r['race_id'] == attempt['race_id'])
        command = original['score_step']['command']
        try:
            native = load_native(Path(command[1]))
        except ModuleNotFoundError as exc:
            self.skipTest('native July dependencies unavailable: ' + str(exc))
        path = lambda flag: Path(command[command.index(flag) + 1])
        context = native._sidecar_context(json.loads(path('--sidecar').read_text()))
        kwargs = {'context':context, 'race_id':attempt['race_id'], 'form_csv_path':path('--form-csv')}
        for flag, name in (('--feature-rows','feature_rows'),('--feature-manifest','feature_manifest'),('--implementation-manifest','implementation_manifest')):
            p = path(flag)
            raw = p.read_bytes()
            kwargs[name + '_path'] = p
            kwargs[name + '_sha'] = hashlib.sha256(raw).hexdigest()
            kwargs[name if name == 'implementation_manifest' else name + '_raw'] = json.loads(raw) if name == 'implementation_manifest' else raw
        with self.assertRaisesRegex(native.ManualPredictionError, 'feature_row_target_grade_mismatch'):
            native._feature_packet(**kwargs)
        before = dict(context)
        result = corrected_feature_packet(native, kwargs, aliases())
        self.assertEqual(len(result[0]), 6)
        self.assertEqual(context, before)
        damaged = dict(kwargs, feature_rows_sha='0' * 64)
        with self.assertRaisesRegex(native.ManualPredictionError, 'feature_rows_manifest_hash_mismatch'):
            corrected_feature_packet(native, damaged, aliases())
