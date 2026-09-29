"""Access boundary and decision-time regression tests using synthetic rows."""
import json
from datetime import datetime, timedelta, timezone

import unittest

from scripts.audit_retained_market_movement import field_complete, scoped_rows, scalar, timing_pair, timestamp


class MarketMovementQualificationTests(unittest.TestCase):
    def test_protected_record_is_skipped_before_full_decode(self):
        payload = b'{"race_id":"protected", "secret_label": THIS IS NOT JSON}\n'
        assert list(scoped_rows(payload, {('allowed',1): 'DOG'})) == []


    def test_allowed_race_unadmitted_runner_rejected_before_payload_decode(self):
        payload = b'{"race_id":"allowed", "box_number":2,"label": NOT JSON}\n'
        with self.assertRaisesRegex(ValueError, 'unadmitted runner'):
            list(scoped_rows(payload, {('allowed',1): 'DOG'}))


    def test_exact_admitted_runner_can_be_decoded(self):
        row = {'race_id':'allowed','box_number':1,'capture_timestamp':'2026-06-10T10:00:00+10:00'}
        assert list(scoped_rows(json.dumps(row).encode(), {('allowed',1):'DOG'})) == [row]


    def test_ambiguous_race_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            scalar('{"race_id":"allowed","nested":{"race_id":"protected"}}','race_id')


    def test_timezones_are_required(self):
        with self.assertRaisesRegex(ValueError, 'timezone'):
            timestamp('2026-06-10T10:00:00')


    def test_cutoff_uses_latest_predecision_quote_and_earliest_qualified_early(self):
        jump = datetime(2026,6,10,10,tzinfo=timezone.utc)
        t = lambda m: jump-timedelta(minutes=m)
        assert timing_pair({t(30),t(20),t(10),t(3),t(1)},jump) == (t(30),t(3))


    def test_age_spacing_and_early_window_boundaries(self):
        jump = datetime(2026,6,10,10,tzinfo=timezone.utc)
        t = lambda m: jump-timedelta(minutes=m)
        assert timing_pair({t(30),t(10)},jump) == (t(30),t(10))
        assert timing_pair({t(30.01),t(10)},jump) is None
        assert timing_pair({t(30),t(10.01)},jump) is None
        assert timing_pair({t(3.99),t(2)},jump) is None
        assert timing_pair({t(4),t(2)},jump) == (t(4),t(2))


    def test_duplicate_box_cannot_look_like_complete_field(self):
        assert not field_complete([{'box_number':1},{'box_number':1},{'box_number':2}],{1,2})
        assert not field_complete([{'box_number':1}],{1,2})
        assert field_complete([{'box_number':2},{'box_number':1}],{1,2})


if __name__ == "__main__":
    unittest.main()
