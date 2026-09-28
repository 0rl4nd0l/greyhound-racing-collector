"""Synthetic tests; no historical or prospective result files are opened."""
import json
import unittest

from scripts.audit_early_speed_neighbours import (
    freeze_sample, identities, key, neighbours, parse_number, scalar,
)


class AuditTests(unittest.TestCase):
    def identity(self, rid='Race 1 - TEST - 2026-06-10'):
        # The result payload deliberately is invalid JSON. Projection must never
        # decode a complete row, even for an admitted identity.
        return ('{"race_id":' + json.dumps(rid) + ',"race_date":' +
                json.dumps(rid.rsplit(' - ', 1)[1]) +
                ',"box":1,"dog_token":"DOG","y":FORBIDDEN_PAYLOAD}').encode()

    def test_identity_projection_never_decodes_result(self):
        self.assertEqual(identities(self.identity(), {}, {})[0]['dog_token'], 'DOG')

    def test_protected_and_out_of_scope_rejected(self):
        rid = 'Race 1 - TEST - 2026-06-10'
        for payload, denied, incident in [
            (self.identity(), {key(rid)}, set()),
            (self.identity(), set(), {rid}),
            (self.identity('Race 1 - TEST - 2026-07-15'), set(), set()),
        ]:
            with self.assertRaisesRegex(ValueError, 'ineligible'):
                identities(payload, denied, incident)

    def test_nested_repeated_identity_must_agree(self):
        self.assertEqual(scalar('{"race_id":"A","source":{"race_id":"A"}}', 'race_id'), 'A')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            scalar('{"race_id":"A","source":{"race_id":"B"}}', 'race_id')

    def test_identity_day_mismatch_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            identities(self.identity().replace(b'"race_date":"2026-06-10"', b'"race_date":"2026-06-11"'), {}, {})
        with self.assertRaises(ValueError):
            identities(self.identity() + b'\n' + self.identity(), {}, {})

    def test_missing_nonfinite_and_zero_never_become_speed(self):
        for value, expected in [(None, 'missing'), ('', 'missing'), ('-', 'malformed'),
                                ('nan', 'nonfinite'), ('inf', 'nonfinite'),
                                ('0', 'nonpositive'), (0, 'nonpositive'), ('-1', 'nonpositive')]:
            self.assertEqual(parse_number(value), (expected, None))
        self.assertEqual(parse_number(' 5.12 '), ('positive', 5.12))

    def test_nearest_occupied_preserves_empty_box_gap(self):
        self.assertEqual(neighbours(3, [1, 3, 6, 8]), [{'box': 1, 'gap': 2}, {'box': 6, 'gap': 3}])
        self.assertEqual(neighbours(1, [1, 3, 6]), [{'box': 3, 'gap': 2}])
        self.assertEqual(neighbours(8, [8]), [])

    def test_sample_only_depends_on_identity_and_is_order_invariant(self):
        ids = ['Race 1 - A - 2026-06-10', 'Race 2 - A - 2026-06-10', 'Race 1 - B - 2026-06-11']
        sample = freeze_sample(ids)
        self.assertEqual(sample, freeze_sample(list(reversed(ids))))
        self.assertEqual(len(sample), 2)
        self.assertEqual({r.split(' - ')[1] for r in sample}, {'A', 'B'})


if __name__ == '__main__':
    unittest.main()
