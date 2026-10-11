"""Failure cases for original evidence authentication and exact result closure."""
import copy
import json

import pytest

from scripts import live_benchmark_alignment as a


def original():
    rid = 'Race 1 - TEST - 2026-07-19'
    rows = [{'race_id': rid, 'runner_id': f'{rid}|box:{box}|dog:{dog}', 'dog_name': dog,
             'box_number': box, 'strict_win_odds': 2.0,
             'market_probability': .5, 'full_probability': .5, 'half_probability': .5,
             'feature_freeze_timestamp': '2026-07-19T10:00:00+10:00',
             'odds_capture_timestamp': '2026-07-19T10:00:00+10:00'}
            for box, dog in [(1, 'ALPHA'), (2, 'BETA')]]
    r = {'schema_version': 'market_form_residual_shadow_record_v1', 'race_id': rid,
         'activation': False, 'outcomes_present': False, 'predictions': rows,
         'score_timestamp': '2026-07-19T10:01:00+10:00', 'jump_timestamp': '2026-07-19T10:05:00+10:00',
         'model_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64,
         'runner_set_sha256': a.digest(('\n'.join(sorted(x['runner_id'] for x in rows)) + '\n').encode())}
    identity = {k: r[k] for k in ('race_id', 'runner_set_sha256', 'model_sha256', 'manifest_sha256')}
    r['record_key'] = a.digest(a.canonical(identity) + b'\n')
    return r


def test_original_identity_and_market_are_checked():
    r = original()
    a.validate_original(r)
    r['model_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='record_key'):
        a.validate_original(r)


@pytest.mark.parametrize('change,reason', [
    (lambda r: r['predictions'][0].update(strict_win_odds=3), 'normalization'),
    (lambda r: r.update(score_timestamp=r['jump_timestamp']), 'chronology'),
    (lambda r: r['predictions'][1].update(box_number=1), 'duplicate'),
])
def test_rejects_corrupt_prediction(change, reason):
    r = original()
    change(r)
    with pytest.raises(ValueError, match=reason):
        a.validate_original(r)


def projection(tmp_path, positions):
    r = original()
    race = {'race_id': r['race_id'], 'status': 'resulted', 'source': 'thedogs_official',
            'race_date': '2026-07-19', 'race_number': 1,
            'source_url': 'https://example.invalid/official', 'box_order': positions,
            'winner_box': 1, 'winner_name': 'ALPHA', 'captured_at': '2026-07-19T10:10:00+10:00'}
    runners = [{'race_id': r['race_id'], 'box_number': box, 'dog_name': 'ALPHA' if box == 1 else 'BETA',
                'finish_position': i + 1, 'is_winner': int(i == 0), 'source': 'thedogs_official',
                'source_url': race['source_url']} for i, box in enumerate(positions)]
    (tmp_path / 'official_result_races.jsonl').write_text(json.dumps(race) + '\n')
    (tmp_path / 'official_result_runners.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in runners))
    def wrap(row):
        return {**row, 'row_json': json.dumps(row), 'source_artifact_dir': str(tmp_path)}
    return r, {'races': [wrap(race)], 'runners': [wrap(x) for x in runners]}


def test_partial_result_never_reconstructs_complete_field(tmp_path):
    r, p = projection(tmp_path, [1])
    joined, error = a.join_result(r, p)
    assert error is None
    assert joined['field_status'] == 'RESULT_FIELD_PARTIAL'
    assert len(r['predictions']) == 2


def test_result_projection_tamper_rejected(tmp_path):
    r, p = projection(tmp_path, [1, 2])
    p['runners'][1]['row_json'] = json.dumps({'tampered': True})
    joined, error = a.join_result(r, p)
    assert joined is None
    assert error == 'RETAINED_RESULT_PROJECTION_MISMATCH'


def test_conflicting_result_snapshots_rejected(tmp_path):
    r, p = projection(tmp_path, [1, 2])
    extra = copy.deepcopy(p['races'][0])
    body = json.loads(extra['row_json'])
    body['box_order'] = [2, 1]
    extra['row_json'] = json.dumps(body)
    p['races'].append(extra)
    assert a.join_result(r, p)[1] == 'CONFLICTING_OFFICIAL_RESULTS'
