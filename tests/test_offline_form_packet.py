"""Safety and temporal tests for the isolated historical research loader."""
from datetime import date
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import offline_form_packet as packet


def test_protected_race_excluded_before_card_read():
    rid = 'Race 1 - AP_K - 2026-06-01'
    content = {
        'development_races.csv': [{'race_id': rid}],
        'development_runners.csv': [{'race_id': rid}],
        'out_of_time_races.csv': [],
    }
    reads = []

    def verified(path, digest, size=None):
        reads.append(path.name)
        assert path.name in {'build_form_only_v1_packet.py', 'development_features.csv'}
        return b''

    with patch.object(packet, '_csv', lambda name: content[name]), patch.object(packet, '_verified', verified):
        rows, audit = packet.load_features(excluded_race_ids=[rid])
    assert rows == []
    assert audit['exclusions'] == [{'race_id': rid, 'reason': 'PROTECTED_IDENTITY'}]
    assert reads == ['build_form_only_v1_packet.py', 'development_features.csv']


def test_future_history_removed_before_recency_and_weighting():
    history, rejected = packet.canonical.accepted_history([
        {'DATE': '2026-06-09', 'PLC': '4', 'MGN': '8'},
        {'DATE': '2026-06-10', 'PLC': '1', 'MGN': '0'},
        {'DATE': '2026-06-11', 'PLC': '1', 'MGN': '0'},
    ], date(2026, 6, 10))
    assert len(history) == 1
    assert packet._weighted(history, 'finish') == 4
    assert [reason for reason, _ in rejected] == ['TARGET_OR_POST_TARGET_HISTORY'] * 2


def test_source_mutation_rejected():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'card.csv'
        source.write_bytes(b'changed')
        with unittest.TestCase().assertRaisesRegex(ValueError, 'source hash mismatch'):
            packet._verified(source, '0' * 64)


def test_distance_uses_explicit_metres_suffix_without_guessing():
    assert packet._metres('400m') == 400
    assert packet._metres(400) == 400
    assert packet._metres('400 m') == 400
    assert packet._metres('400-500m') is None
    assert packet._metres('') is None


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_protected_race_excluded_before_card_read,
        test_future_history_removed_before_recency_and_weighting,
        test_source_mutation_rejected,
        test_distance_uses_explicit_metres_suffix_without_guessing,
    ))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
