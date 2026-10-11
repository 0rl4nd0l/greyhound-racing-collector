import json
from scripts import reconcile_retained_live_results as r
from scripts import live_benchmark_alignment as a
from tests.test_live_benchmark_alignment import projection


def test_mixed_archive_decodes_only_exact_allowed_child():
    raw = b'{"items":[{"race_id":"reserved","secret":"never decode"},{"race_id":"allowed","positions":[1]}]}'
    found = list(r.allowlisted_json_objects(raw, {'allowed'}))
    assert [x[1]['race_id'] for x in found] == ['allowed']
    assert raw[found[0][0]:].startswith(b'{"race_id":"allowed"')


def test_nested_outside_scope_payload_is_not_decoded():
    raw = b'{"race_id":"allowed","child":{"race_id":"reserved","positions":[1]}}'
    assert list(r.allowlisted_json_objects(raw, {'allowed'})) == []
    assert list(r.allowlisted_jsonl(raw, {'allowed'})) == []


def test_losing_ties_preserve_win_target_and_winning_ties_reject():
    rows = [{'box_number': i + 1, 'finish_position': p} for i, p in enumerate([1, 2, 3, 4, 5, 5, 7, 8])]
    assert r.qualified_win_order(list(range(1, 9)), rows)
    rows[1]['finish_position'] = 1
    assert not r.qualified_win_order(list(range(1, 9)), rows)


def test_invalid_rank_gap_and_duplicate_box_reject():
    rows = [{'box_number': 1, 'finish_position': 1}, {'box_number': 2, 'finish_position': 3}]
    assert not r.qualified_win_order([1, 2], rows)
    assert not r.qualified_win_order([1, 1], rows)


def test_circular_result_names_do_not_establish_unchanged_field(tmp_path):
    original, result = projection(tmp_path, [1, 2])
    result['official_runner_identity_observed'] = False
    joined, error = a.join_result(original, result)
    assert error is None
    assert joined['field_status'] == 'RESULT_RUNNER_IDENTITY_UNVERIFIED'


def test_native_nonwinning_tie_join(tmp_path):
    original, result = projection(tmp_path, [1, 2])
    original['predictions'].append({**original['predictions'][1], 'box_number': 3,
                                  'runner_id': original['race_id'] + '|box:3|dog:BETA'})
    body = json.loads(result['races'][0]['row_json'])
    body['box_order'] = [1, 2, 3]
    result['races'][0].update(body, row_json=json.dumps(body))
    third = {**json.loads(result['runners'][1]['row_json']), 'box_number': 3}
    result['runners'].append({**third, 'row_json': json.dumps(third), 'source_artifact_dir': str(tmp_path)})
    (tmp_path / 'official_result_races.jsonl').write_text(json.dumps(body) + '\n')
    (tmp_path / 'official_result_runners.jsonl').write_text(''.join(x['row_json'] + '\n' for x in result['runners']))
    joined, error = a.join_result(original, result)
    assert error is None
    assert joined['winner_box'] == 1
    assert [x['finish_position'] for x in result['runners']] == [1, 2, 2]


def test_correction_append_is_idempotent_across_json_tuple_replay(tmp_path):
    path = tmp_path / 'correction-trail.json'
    event = {'race_id': 'one', 'preserved_finishes': [(1, 1), (2, 2)]}
    r.append_corrections(path, [event])
    first = path.read_bytes()
    r.append_corrections(path, [event])
    assert path.read_bytes() == first
    assert len(json.loads(path.read_bytes())) == 1


def test_existing_duplicate_history_preserved_with_hash(tmp_path):
    path = tmp_path / 'correction-trail.json'
    event = {'race_id': 'one', 'preserved_finishes': [[1, 1], [2, 2]]}
    path.write_text(json.dumps([event, event]))
    original_bytes = path.read_bytes()
    r.append_corrections(path, [event])
    assert len(json.loads(path.read_bytes())) == 1
    audit = json.loads((tmp_path / 'correction-trail-audit.json').read_bytes())
    assert audit['duplicate_append_entries'] == 1
    assert r.sha(original_bytes) == audit['sha256']
    assert r.Path(audit['preserved_original']).read_bytes() == original_bytes
