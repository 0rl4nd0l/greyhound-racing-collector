import unittest
import numpy as np

from scripts.history_support_shrinkage import adjust, race_support, select_parameter, metric_summary
from scripts.offline_form_packet import FEATURES


class SupportTests(unittest.TestCase):
    def test_support_uses_counts_and_missingness_without_labels(self):
        features = dict.fromkeys(FEATURES, 1.0)
        features.update(prior_start_count=5, starts_same_venue=3,
                        starts_same_distance=3, same_grade_start_count=3)
        self.assertEqual(race_support([{'features': features}]), 1)
        features['win_rate_same_venue'] = None
        self.assertAlmostEqual(race_support([{'features': features, 'y': 'not read'}]), .625)
        features['win_rate_same_venue'] = float('nan')
        self.assertAlmostEqual(race_support([{'features': features}]), .625)
        features.update(starts_same_venue=0, starts_same_distance=0, same_grade_start_count=0)
        self.assertEqual(race_support([{'features': features}]), 0)

    def test_incomplete_distributions_cannot_be_renormalized_silently(self):
        for full in ([.2, .2], [0, 1], [float('nan'), .5]):
            with self.assertRaises(ValueError):
                adjust([.5, .5], full, .5)

    def test_earlier_selection_prefers_less_shrinkage_when_full_better(self):
        features = dict.fromkeys(FEATURES, 1.0)
        features.update(prior_start_count=5, starts_same_venue=3,
                        starts_same_distance=3, same_grade_start_count=3)
        rows = [{'race_id': 'earlier', 'box': i+1, 'features': features,
                 'market': .5, 'full': p, 'y': int(i == 0)} for i, p in enumerate([.6, .4])]
        chosen, trials = select_parameter(rows, [.25, 1])
        self.assertEqual(chosen, .25)
        self.assertEqual(len(trials), 2)
        # No evaluation outcomes are accepted by this interface.
        for row in rows:
            row['full'] = .5
        self.assertEqual(select_parameter(rows, [.25, 1])[0], 1)

    def test_date_cluster_lodo_uses_race_weighted_pairing(self):
        races = []
        for i, (day, delta) in enumerate([('a', -.1), ('a', -.1), ('b', .2), ('b', .2), ('c', -.1), ('c', -.1)]):
            row = {'race_id': str(i), 'race_date': day, 'outer': 'period', 'support': .5}
            for name in ('market', 'full', 'half'):
                row[name] = {'ll': 1., 'brier': 1., 'accuracy': .5}
            row['adaptive'] = {'ll': 1.+delta, 'brier': 1.+delta, 'accuracy': .5}
            races.append(row)
        summary = metric_summary(races, draws=100)
        paired = summary['paired'][0]
        self.assertAlmostEqual(paired['difference_adaptive_minus_baseline'], 0)
        self.assertAlmostEqual(paired['leave_one_date_out']['b'], -.1)
        self.assertEqual(paired['improved_races'], 4)
        self.assertEqual(paired['harmed_dates'], 1)

    def test_support_zero_returns_market_and_constant_half_reproduces_half(self):
        market = np.array([.6, .4])
        full = np.array([.5, .5])
        np.testing.assert_allclose(adjust(market, full, 0), market)
        expected = np.array([np.sqrt(.3), np.sqrt(.2)])
        expected /= expected.sum()
        np.testing.assert_allclose(adjust(market, full, .5), expected)
        np.testing.assert_allclose(adjust(market, full, 1), full)


if __name__ == '__main__':
    unittest.main()
