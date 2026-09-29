import json
from pathlib import Path
import tempfile
import unittest

from scripts.offline_form_packet import FEATURES
from scripts.offline_systematic_search import Ledger
from scripts.run_history_depth_comparison import admitted_pairs, fit_receipt, paired_summary


class HistoryDepthComparisonTests(unittest.TestCase):
    def pairs(self):
        allowed = {('race', 1): 'a', ('race', 2): 'b'}
        features = dict.fromkeys(FEATURES, None)
        original = {key: {'features': dict(features)} for key in allowed}
        pairs = [{'race_id': rid, 'box': box, 'short_features': dict(features), 'richer_features': dict(features)} for rid, box in allowed]
        return allowed, original, pairs

    def test_complete_fields_and_original_short_values_are_required(self):
        allowed, original, pairs = self.pairs()
        payload = lambda rows: '\n'.join(json.dumps(r) for r in rows).encode()
        self.assertEqual(len(admitted_pairs(payload(pairs), allowed, original)), 2)
        with self.assertRaisesRegex(ValueError, 'incomplete history field'):
            admitted_pairs(payload(pairs[:1]), allowed, original)
        pairs[0]['short_features']['career_win_rate'] = 0.0
        with self.assertRaisesRegex(ValueError, 'short features changed'):
            admitted_pairs(payload(pairs), allowed, original)

    def test_unadmitted_identity_is_rejected_before_feature_decode(self):
        allowed, original, _ = self.pairs()
        payload = b'{"race_id":"denied","box":1,"payload":not-valid-json}'
        with self.assertRaisesRegex(ValueError, 'unadmitted'):
            admitted_pairs(payload, allowed, original)

    def test_duplicate_runner_rejected(self):
        allowed, original, pairs = self.pairs()
        payload = '\n'.join(json.dumps(r) for r in pairs + pairs[:1]).encode()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            admitted_pairs(payload, allowed, original)

    def test_training_cutoff_failure_retained_without_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = Ledger(root / 'ledger.jsonl')
            with self.assertRaisesRegex(ValueError, 'training chronology'):
                fit_receipt([{'race_date': '2026-07-01'}], [], root / 'attempt', {}, {}, {}, ledger,
                    'short', {'name': 'fold', 'test_start': '2026-07-01', 'test_end': '2026-07-02'})
            events = [json.loads(line) for line in (root / 'attempt/attempt.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events], ['START', 'FAILED'])
            self.assertFalse((root / 'attempt/receipt.json').exists())

    def test_paired_date_bootstrap_keeps_races_and_reports_harm(self):
        races = []
        for i in range(12):
            baseline = {'ll': 1.0, 'brier': .5, 'accuracy': 0.0}
            richer = {'ll': 1.1, 'brier': .52, 'accuracy': 0.0}
            races.append({'race_id': str(i), 'race_date': '2026-07-0' + str(i % 3 + 1), 'outer': 'period',
                          'runners': 6, 'enriched_runners': i % 2, 'market': baseline, 'short': baseline, 'richer': richer})
        result = paired_summary(races, draws=100)
        contrast = result['paired'][0]
        self.assertAlmostEqual(contrast['mean_difference'], .1)
        self.assertAlmostEqual(contrast['pointwise95'][0], .1)
        self.assertEqual(contrast['harmed_races'], 12)
        self.assertEqual(contrast['harmed_dates'], 3)
        self.assertEqual(result['overall']['runners'], 72)


if __name__ == '__main__':
    unittest.main()
