import json
from pathlib import Path
import pytest
from scripts import project_acquired_live_results as p
from scripts import live_benchmark_alignment as alignment

URL = 'https://www.thedogs.com.au/racing/healesville/2026-07-17/7/db-construction?trial=false'
RID = 'Race 7 - HEA - 2026-07-17'


def markup(rows, url=URL):
    body = ''.join(f'<tr class="race-runner"><td class="race-runners__finish-position">{place}</td>'
                   f'<td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td>'
                   f'<td class="race-runners__name">{name}</td></tr>' for box, name, place in rows)
    return f'<link rel="canonical" href="{url}"><table class="race-runners--result">{body}</table>'.encode()


def original():
    return {'race_id': RID, 'jump_timestamp': '2026-07-17T12:52:00+10:00',
            'predictions': [{'box_number': 1, 'dog_name': 'Alpha', 'runner_id': RID + '|box:1|dog:ALPHA'},
                            {'box_number': 2, 'dog_name': 'Beta', 'runner_id': RID + '|box:2|dog:BETA'}]}


def item(tmp_path, raw):
    response = tmp_path / 'response.html'
    response.write_bytes(raw)
    meta = {'requested_url': URL, 'final_url': URL, 'http_status': 200,
            'acquired_at': '2026-10-11T12:00:00+11:00', 'response_sha256': p.digest(raw)}
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps(meta))
    return {'race_id': RID, 'response_path': str(response), 'metadata_path': str(metadata)}


def test_native_names_and_hash_bound_projection(tmp_path):
    rows = [(1, 'Alpha', '1st'), (2, 'Beta', '2nd'), (9, 'Unused reserve', 'SCR')]
    projected = p.project_one(item(tmp_path, markup(rows)), original(), URL, tmp_path / 'derived')
    assert projected['official_runner_identity_observed'] is True
    assert projected['runner_name_origin'] == 'SOURCE_NATIVE_OFFICIAL_HTML'
    assert [r['dog_name'] for r in projected['runners']] == ['Alpha', 'Beta']
    assert projected['races'][0]['unused_scratched_reserves'][0]['box_number'] == 9
    joined, error = alignment.join_result(original(), projected)
    assert error is None
    assert joined['field_status'] == 'EXACT_UNCHANGED'


def test_explicit_dnf_qualifies_without_invented_finishing_position(tmp_path):
    projected = p.project_one(item(tmp_path, markup([(1, 'Alpha', '1st'), (2, 'Beta', 'DNF')])), original(), URL, tmp_path / 'derived')
    assert projected['runners'][1]['finish_position'] is None
    assert projected['runners'][1]['terminal_status'] == 'DNF'
    joined, error = alignment.join_result(original(), projected)
    assert error is None
    assert joined['field_status'] == 'EXACT_UNCHANGED'


@pytest.mark.parametrize('rows,reason', [
    ([(1, 'Alpha', '1st'), (2, 'Replacement', '2nd')], 'name_changed'),
    ([(1, 'Alpha', '1st'), (2, 'Beta', 'SCR')], 'runner_scratched'),
    ([(1, 'Alpha', '1st')], 'field_incomplete'),
    ([(1, 'Alpha', '1st'), (2, '', '2nd')], 'name_missing'),
    ([(1, 'Alpha', '1st'), (2, 'Beta', '')], 'terminal_status_missing'),
    ([(1, 'Alpha', '1st'), (2, 'Beta', '1st')], 'win_target_ambiguous'),
    ([(1, 'Alpha', '1st'), (2, 'Beta', '2nd'), (2, 'Beta', '2nd')], 'duplicate_official_box'),
])
def test_unusable_fields_never_imputed(tmp_path, rows, reason):
    with pytest.raises(ValueError, match=reason):
        p.project_one(item(tmp_path, markup(rows)), original(), URL, tmp_path / 'derived')


def test_source_identity_and_byte_authentication(tmp_path):
    payload = markup([(1, 'Alpha', '1st'), (2, 'Beta', '2nd')], URL.replace('/7/', '/8/'))
    with pytest.raises(ValueError, match='source_declared_race_identity'):
        p.project_one(item(tmp_path, payload), original(), URL, tmp_path / 'derived')
    response = item(tmp_path, markup([(1, 'Alpha', '1st'), (2, 'Beta', '2nd')]))
    Path(response['response_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='response_hash_mismatch'):
        p.project_one(response, original(), URL, tmp_path / 'derived')


def test_source_derived_names_are_not_replaced_from_forecast(tmp_path):
    projected = p.project_one(item(tmp_path, markup([(1, 'ALPHA', '1st'), (2, 'Beta', 'FELL')])), original(), URL, tmp_path / 'derived')
    assert projected['runners'][0]['dog_name'] == 'ALPHA'
    assert projected['runners'][1]['finish_position'] is None


def test_valid_reserve_note_maps_existing_forecast_identity(tmp_path):
    payload = markup([(1, 'Alpha', '1st'), (2, 'Old runner', 'SCR'), (9, 'Beta (from box 2)', '2nd')])
    projected = p.project_one(item(tmp_path, payload), original(), URL, tmp_path / 'derived')
    assert projected['runners'][1]['original_source_box'] == 9
    assert projected['runners'][1]['box_number'] == 2
    assert projected['races'][0]['reserve_mapping']['ignored_terminal_status_rows'][0]['dog_name'] == 'Old runner'


@pytest.mark.parametrize('field,value', [('requested_url', URL.replace('?trial=false', '')), ('final_url', URL + '&redirected=true')])
def test_exact_authorized_request_and_no_redirect_required(tmp_path, field, value):
    request = item(tmp_path, markup([(1, 'Alpha', '1st'), (2, 'Beta', '2nd')]))
    metadata = json.loads(Path(request['metadata_path']).read_bytes())
    metadata[field] = value
    Path(request['metadata_path']).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='exact_authorized_request_or_redirect'):
        p.project_one(request, original(), URL, tmp_path / 'derived')


def test_derived_witness_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'derived.json'
    p.write(path, {'value': 1})
    previous = path.read_bytes()
    p.write(path, {'value': 1})
    with pytest.raises(ValueError, match='already_exists_with_different_bytes'):
        p.write(path, {'value': 2})
    assert path.read_bytes() == previous


def test_nonwinning_tie_stays_source_ranked(tmp_path):
    source = original()
    source['predictions'].append({'box_number': 3, 'dog_name': 'Gamma', 'runner_id': RID + '|box:3|dog:GAMMA'})
    request = item(tmp_path, markup([(1, 'Alpha', '1st'), (2, 'Beta', '2nd'), (3, 'Gamma', '2nd')]))
    projected = p.project_one(request, source, URL, tmp_path / 'derived')
    assert [r['finish_position'] for r in projected['runners']] == [1, 2, 2]
    assert alignment.join_result(source, projected)[1] is None


def test_unparsed_extra_source_runner_is_never_silently_dropped(tmp_path):
    raw = markup([(1, 'Alpha', '1st'), (2, 'Beta', '2nd')]).replace(b'</table>', b'<tr class="race-runner"><td class="race-runners__name">Unexpected</td></tr></table>')
    with pytest.raises(ValueError, match='runner_row_not_parsed'):
        p.project_one(item(tmp_path, raw), original(), URL, tmp_path / 'derived')


def test_canonical_trial_true_is_outside_race_scope(tmp_path):
    raw = markup([(1, 'Alpha', '1st'), (2, 'Beta', '2nd')], URL.replace('trial=false', 'trial=true'))
    with pytest.raises(ValueError, match='trial_scope_mismatch'):
        p.project_one(item(tmp_path, raw), original(), URL, tmp_path / 'derived')
