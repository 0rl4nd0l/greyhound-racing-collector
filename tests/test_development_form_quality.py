import unittest
import json
import tempfile
from datetime import date
from pathlib import Path
import numpy as np

from scripts.development_form_quality import build_record
from scripts.development_form_fit import fit_diagnostic


def raw(day, finish=2, **fields):
    return {'DATE': day, 'PLC': str(finish), 'TRACK': 'GEE', 'DIST': '400',
            'G': '5', 'BOX': '1', 'MGN': '2.5', **fields}


def record(rows, **target):
    return build_record(rows, target_date=date(2026, 6, 10),
                        venue=target.get('venue', 'GEE'),
                        distance=target.get('distance', 400),
                        grade=target.get('grade', 'GRADE_5'))


class QualityTests(unittest.TestCase):
    def test_source_paths_are_selected_only_from_admitted_identities(self):
        from scripts.audit_development_form_quality import admitted_sources
        allowed = {('good', 1): 'dog'}
        source = {'race_id': 'good', 'card_source_path': '/admitted'}
        provenance = {'sources': [source, {'race_id': 'protected', 'card_source_path': '/do-not-open'}]}
        self.assertEqual(admitted_sources(provenance, allowed), {'good': source})
        with self.assertRaisesRegex(ValueError, 'missing admitted'):
            admitted_sources({'sources': provenance['sources'][1:]}, allowed)
        with self.assertRaisesRegex(ValueError, 'duplicate admitted'):
            admitted_sources({'sources': [source, source]}, allowed)

    def test_failed_chronology_is_retained_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'failed'
            with self.assertRaisesRegex(ValueError, 'training must precede'):
                fit_diagnostic([], out=out, training_end='2026-06-03',
                               evaluation_start='2026-06-03', admission={'fixture': True})
            events = [json.loads(line)['event'] for line in (out / 'attempt.jsonl').read_text().splitlines()]
            self.assertEqual(events, ['START', 'FAILED'])
            self.assertFalse((out / 'receipt.json').exists())

    def test_new_diagnostic_fit_receipt_replays_and_never_overwrites(self):
        from scripts.development_form_quality import NAMES
        from scripts.offline_systematic_search import predict
        rows = []
        for race, day in [('a', '2026-06-01'), ('b', '2026-06-02')]:
            for box in (1, 2):
                rows.append({'race_id': race, 'race_date': day, 'box': box,
                             'dog_token': f'{race}{box}', 'y': int(box == 1),
                             'market': .5, 'features': dict.fromkeys(NAMES.values(), float(box))})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fit'
            result = fit_diagnostic(rows, out=out, training_end='2026-06-02',
                                    evaluation_start='2026-06-03', admission={'fixture': True})
            receipt = json.loads((out / 'receipt.json').read_text())
            retained = json.loads((out / 'training_rows.json').read_text())
            model = receipt['model']
            model['beta'] = np.array(model['beta'])
            for key in ('median', 'mean', 'scale'):
                model['prep'][key] = np.array(model['prep'][key])
            np.testing.assert_allclose(predict(retained, model), result, atol=1e-14)
            self.assertEqual(len(receipt['training_membership']), 4)
            self.assertEqual(len(receipt['model']['beta']), 32)
            self.assertTrue(receipt['environment']['executable_sha256'])
            self.assertEqual(receipt['identity'], 'new_diagnostic_fit_not_original_artifact')
            with self.assertRaises(FileExistsError):
                fit_diagnostic(rows, out=out, training_end='2026-06-02',
                               evaluation_start='2026-06-03', admission={'fixture': True})

    def test_recent_and_retained_differ_only_with_older_observations(self):
        rows = [raw(f'2026-06-0{i}', finish=2) for i in range(4, 10)]
        rows[0]['PLC'] = '1'
        got = record(rows)
        self.assertEqual(got['features']['recent_win_rate_5'], 0)
        self.assertAlmostEqual(got['features']['retained_win_rate'], 1 / 6, places=7)
        self.assertEqual(got['quality']['retained_win_rate']['observed_denominator'], 6)
        self.assertEqual(got['history']['career_completeness'], 'unknown')
        self.assertNotIn('career_win_rate', got['features'])

    def test_partial_context_does_not_default_to_zero(self):
        got = record([raw('2026-06-09', DIST='')])
        self.assertIsNone(got['features']['retained_exact_distance_start_count'])
        self.assertEqual(got['quality']['retained_exact_distance_start_count']['status'],
                         'history_context_incomplete')
        self.assertEqual(got['features']['retained_same_venue_win_rate'], 0)

    def test_denominator_and_recorded_margin_do_not_imply_beaten_margin(self):
        got = record([raw('2026-06-09', 1, MGN='3'), raw('2026-06-08', PLC='', MGN='')])
        self.assertEqual(got['features']['retained_start_count'], 2)
        self.assertEqual(got['features']['retained_win_rate'], 1)
        self.assertEqual(got['quality']['retained_win_rate']['observed_denominator'], 1)
        self.assertEqual(got['features']['recent_recorded_margin_mean_5'], 3)
        self.assertEqual(got['quality']['recent_recorded_margin_mean_5']['observed_denominator'], 1)

    def test_date_dedup_and_cap_preserve_source_limits(self):
        rows = [raw(f'2026-05-{i:02d}') for i in range(1, 23)]
        rows += [rows[-1], raw('2026-06-10')]
        got = record(rows)
        self.assertEqual(got['history']['accepted_rows'], 20)
        self.assertEqual(got['history']['rejections'], {
            'NORMALIZED_DUPLICATE_HISTORY': 1, 'TARGET_OR_POST_TARGET_HISTORY': 1,
            'HISTORY_CAP_20': 2})
        self.assertEqual(got['history']['oldest_date'], '2026-05-03')
        self.assertTrue(got['history']['cap_reached'])

    def test_empty_history_and_invalid_finish_are_not_observed_zero_rates(self):
        got = record([])
        self.assertEqual(got['features']['retained_start_count'], 0)
        self.assertIsNone(got['features']['retained_win_rate'])
        self.assertIsNone(got['features']['retained_same_grade_label_start_count'])
        with self.assertRaisesRegex(ValueError, 'invalid recorded finish'):
            record([raw('2026-06-09', 0)])

    def test_unknown_context_is_not_zero_observed_matches(self):
        known = record([raw('2026-06-09')], distance=500)
        unknown = record([raw('2026-06-09')], distance=None)
        self.assertEqual(known['features']['retained_exact_distance_start_count'], 0)
        self.assertIsNone(known['features']['retained_exact_distance_win_rate'])
        self.assertEqual(known['quality']['retained_exact_distance_win_rate']['status'],
                         'no_matching_retained_starts')
        self.assertIsNone(unknown['features']['retained_exact_distance_start_count'])
        self.assertEqual(unknown['quality']['retained_exact_distance_start_count']['status'],
                         'target_context_unavailable')


if __name__ == '__main__':
    unittest.main()
