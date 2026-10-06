"""Automatic closure tests use fabricated native bundles and official SQLite rows."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from race_collection import prospective_speed_closure as closure
from race_collection import prospective_speed_runtime as runtime
from tests.test_prospective_speed_evaluation_io import make_fixture


def load(ref):
    return json.loads(Path(ref['path']).read_bytes())


def replace(ref, value):
    """Fixture construction only: recompute bindings after changing fabricated inputs."""
    Path(ref['path']).write_bytes(runtime.encoded(value))
    return runtime.reference(Path(ref['path']))


def fixture(tmp_path, monkeypatch, *, queue_state='CLOSED', dead_heat=False, wrong_name=False):
    base = make_fixture(tmp_path, monkeypatch)
    job = load(base['job'])
    population_ref = job['populations'][0]
    population = load(population_ref)
    race_id = population['selected_race_ids'][0]
    key = hashlib.sha256(race_id.encode()).hexdigest()
    # The coordinator has one global attempts directory.
    (tmp_path/'forecasts').rename(tmp_path/'attempts')
    (tmp_path/'attempts').chmod(0o700)
    attempt = tmp_path/'attempts'/key

    def put(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return runtime.put_new(path, value)

    native_root = tmp_path/'native-bundles'/'job-fixture'
    request = {'job_id': 'job-fixture', 'race_id': race_id,
        'jump_timestamp': '2026-10-10T13:10:00+11:00',
        'runners': [{'box_number': box, 'display_name': name,
            'source_native_runner_id': f'native:dog{box}'}
            for box, name in [(1, 'DOG ONE'), (2, 'DOG TWO')]]}
    request_ref = put('native-bundles/job-fixture/request.json', request)
    race = {'race_id': race_id, 'race_date': '2026-10-10', 'race_number': 1,
        'venue': 'Richmond', 'url': 'https://www.thedogs.com.au/racing/richmond/2026-10-10/1/test'}
    result_ref = put('native-bundles/job-fixture/result.json', {'job_id': 'job-fixture',
        'race': race, 'generated_at': '2026-10-10T13:04:00+11:00'})
    manifest_ref = put('native-bundles/job-fixture/bundle-manifest.json', {'job_id': 'job-fixture',
        'files': {'request.json': {'sha256': request_ref['sha256']},
                  'result.json': {'sha256': result_ref['sha256']}}})
    member = {'race_id': race_id, 'jump_at': request['jump_timestamp'], 'bundle_manifest': manifest_ref}
    source_ref = runtime.reference(tmp_path/'source-job.json')
    source_job = load(source_ref)
    source_job['member'] = member
    source_ref = replace(source_ref, source_job)
    execution = replace(runtime.reference(attempt/'job.json'), {**source_job, 'execution_mode': 'PROSPECTIVE'})
    claim = load(runtime.reference(attempt/'claim.json'))
    claim['source_job'] = source_ref
    claim_ref = replace(runtime.reference(attempt/'claim.json'), claim)
    payload = load(runtime.reference(attempt/'forecast.json'))
    payload['source_member'] = member
    for i, row in enumerate(payload['predictions'], 1):
        row['box_number'] = i
    payload_ref = replace(runtime.reference(attempt/'forecast.json'), payload)
    terminal = load(runtime.reference(attempt/'terminal.json'))
    terminal.update(claim=claim_ref, execution_job=execution, payload=payload_ref)
    terminal_ref = replace(runtime.reference(attempt/'terminal.json'), terminal)
    seal = load(runtime.reference(attempt/'seal.json'))
    seal.update(terminal=terminal_ref, payload=payload_ref)
    seal_ref = replace(runtime.reference(attempt/'seal.json'), seal)
    completion = load(runtime.reference(attempt/'completion.json'))
    completion['seal'] = seal_ref
    completion_ref = replace(runtime.reference(attempt/'completion.json'), completion)
    put('2026-10-10/date-accounting.json', {'local_date': '2026-10-10',
        'status': 'POPULATION_FROZEN', 'population_sha256': closure.planning._digest(population),
        'population': population_ref})
    put('2026-10-10/jobs/'+key+'.disposition.json', {'race_id': race_id,
        'status': 'SEALED_PREJUMP', 'completion': completion_ref})
    put('2026-10-11/date-accounting.json', {'local_date': '2026-10-11', 'status': 'INDEX_MISSING',
        'reason': 'Fabricated missing index', 'population_sha256': None})
    retention = tmp_path/'retention'
    retention.mkdir(mode=0o700)
    queue = retention/'queue.sqlite3'
    with sqlite3.connect(queue) as db:
        db.execute('CREATE TABLE jobs (race TEXT PRIMARY KEY, job TEXT, jump TEXT, state TEXT)')
        if queue_state:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?)',
                (race_id, 'job-fixture', request['jump_timestamp'], queue_state))
        db.execute('INSERT INTO jobs VALUES (?,?,?,?)', ('protected', 'protected', 'invalid', 'CLOSED'))
    database = retention/'results.sqlite3'
    common = {key: value for key, value in race.items() if key != 'url'} | {
        'source': 'thedogs_official', 'source_url': race['url'], 'captured_at': '2026-10-10T14:00:00+11:00'}
    official = {**common, 'status': 'resulted', 'start_datetime': request['jump_timestamp'],
        'winner_box': 1, 'winner_name': 'DOG ONE', 'position_count': 2, 'participant_count': 2,
        'box_order': [1, 2]}
    rows = [{**common, 'box_number': box, 'finish_position': 1 if dead_heat else box,
        'is_winner': box == 1 or dead_heat, 'dog_name': 'WRONG DOG' if wrong_name and box == 2 else name,
        'source_native_runner_id': f'native:dog{box}'} for box, name in [(1, 'DOG ONE'), (2, 'DOG TWO')]]
    with sqlite3.connect(database) as db:
        for table, values in [('races', [official]), ('runners', rows)]:
            table = 'autonomous_official_result_evidence_'+table
            db.execute(f'CREATE TABLE {table} (race_id TEXT, row_json TEXT)')
            for value in values:
                db.execute(f'INSERT INTO {table} VALUES (?,?)', (race_id, json.dumps(value)))
            # This is deliberately undecodable. A whole-table read would fail.
            db.execute(f'INSERT INTO {table} VALUES (?,?)', ('protected', 'NEVER DECODE PROTECTED TARGET'))
    authority_ref = put('retention/authority.json', {'status': 'FABRICATED_AUTHORIZED_EXISTING_RETENTION'})
    sources = {day: {'root': str(retention), 'queue_database': str(queue),
        'result_database': str(database), 'authority': authority_ref} for day in closure.planning.DATES}
    activation = put('closure-activation.json', {'status': 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT',
        'plan_sha256': job['plan']['sha256'], 'development_precedence_verified': True,
        'result_retention_routing_verified': True, 'automatic_result_projection_authorized': True,
        'retention_by_date': sources, 'additional_source_requests': 0, 'additional_result_requests': 0})
    config = put('closure-config.json', {'schema_version': 'prospective_speed_closure_configuration_v1',
        'status': 'AUTHORIZED_PROSPECTIVE_CLOSURE_CONSUMER', 'authority_reference': 'FABRICATED:approval',
        'plan': job['plan'], 'activation': activation, 'retention_by_date': sources,
        'coordinator_state_root': str(tmp_path), 'forecast_root': str(tmp_path/'attempts'),
        'closure_root': str(tmp_path/'automatic-closure')})
    return {'config': config, 'clock': base['clock'], 'root': tmp_path/'automatic-closure',
        'queue': queue, 'database': database, 'attempt': attempt, 'race_id': race_id}


def report(f):
    return json.loads((f['root']/'evaluation'/'evaluation.private.json').read_bytes())


def test_complete_native_sqlite_closure_builds_job_and_scores_once(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    value = closure.finalize(f['config'])
    assert value['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert report(f)['accounting']['scored_races'] == 1
    assert len(report(f)['date_accounting']) == 2
    frozen = (f['root']/'evaluation'/'evaluation.private.json').read_bytes()
    f['database'].unlink()
    again = closure.finalize(f['config'])
    assert again['status'] == 'CLOSURE_LOOK_ALREADY_CONSUMED'
    assert again['result_accesses_consumed'] == 0
    assert frozen == (f['root']/'evaluation'/'evaluation.private.json').read_bytes()


def test_official_dead_heat_uses_prespecified_equal_mass_target(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, dead_heat=True)
    assert closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    entry = json.loads((f['root']/'result-manifest.json').read_bytes())['entries'][0]
    assert load(entry['target'])['outcome'] == [.5, .5]


@pytest.mark.parametrize('queue_state,wrong_name', [(None, False), ('QUARANTINED', False), ('CLOSED', True)])
def test_missing_quarantined_and_wrong_identity_never_score(tmp_path, monkeypatch, queue_state, wrong_name):
    f = fixture(tmp_path, monkeypatch, queue_state=queue_state, wrong_name=wrong_name)
    if queue_state != 'CLOSED':
        f['database'].unlink()  # Must not be needed for a missing/quarantined selected queue record.
    assert closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    value = report(f)
    assert value['accounting']['selected_races'] == 1
    assert value['accounting']['scored_races'] == 0
    entry = json.loads((f['root']/'result-manifest.json').read_bytes())['entries'][0]
    assert entry['target'] is None


def test_fixed_clock_and_authority_gates_precede_result_access(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:04:59+11:00')
    monkeypatch.setattr(closure, '_queue_record', lambda *args: pytest.fail('early result access'))
    assert closure.finalize(f['config'])['result_accesses_consumed'] == 0
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:05:00+11:00')
    config = load(f['config'])
    activation = load(config['activation'])
    activation['automatic_result_projection_authorized'] = False
    config['activation'] = replace(config['activation'], activation)
    f['config'] = replace(f['config'], config)
    with pytest.raises(closure.ClosureRejected, match='AUTHORITY_REQUIRED'):
        closure.finalize(f['config'])


def test_interrupted_preparation_preserved_then_new_outcome_free_attempt(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    previous = f['root']/'prepared-0001'
    f['root'].mkdir(mode=0o700)
    previous.mkdir(parents=True, mode=0o700)
    (previous/'partial').write_bytes(b'preserve')
    assert closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert (previous/'partial').read_bytes() == b'preserve'
    assert (f['root']/'prepared-0002'/'membership-before-results.json').exists()


def test_claim_exists_before_source_and_failure_consumes_look(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    def interrupted(*args):
        assert (f['root']/'closure-claim.json').exists()
        raise RuntimeError('fabricated crash after source access begins')
    monkeypatch.setattr(closure, '_closure', interrupted)
    assert closure.finalize(f['config'])['status'] == 'FAILED_CLOSURE_LOOK_NO_RETRY'
    assert closure.finalize(f['config'])['result_accesses_consumed'] == 0


def test_absent_scheduler_disposition_reconstructed_without_new_forecast(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    key = hashlib.sha256(f['race_id'].encode()).hexdigest()
    (tmp_path/'2026-10-10'/'jobs'/(key+'.disposition.json')).unlink()
    assert closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert report(f)['accounting']['scored_races'] == 1


def test_corrupt_outer_claim_is_consumed_without_reading_sources(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f['root'].mkdir(mode=0o700)
    (f['root']/'closure-claim.json').write_bytes(b'{"config":')
    monkeypatch.setattr(closure, '_queue_record', lambda *args: pytest.fail('consumed result access'))
    value = closure.finalize(f['config'])
    assert value['status'] == 'CLOSURE_LOOK_ALREADY_CONSUMED'
    assert (f['root']/'closure-claim.json').read_bytes() == b'{"config":'


def test_known_nonfinish_reconstructed_from_exact_retained_source_bytes(tmp_path, monkeypatch):
    from scripts.reconcile_comparison_result_identity import RetainedDeadline
    from src.predictor.comparison_terminal_results import known_nonfinish_evidence

    f = fixture(tmp_path, monkeypatch, queue_state='CLOSED_NON_FINISH')
    payload = load(runtime.reference(f['attempt']/'forecast.json'))
    context = closure._native_context(closure.evaluation._Checked(), payload)
    captured = datetime.fromisoformat('2026-10-10T14:00:00+11:00')
    url = context['bundle'].result['race']['url']
    body = (b'<table class="race-runners--result">'
        b'<tr class="race-runner"><td class="race-runners__finish-position">1st</td>'
        b'<td class="race-runners__box"><sprite-svg name="rug_1"></sprite-svg></td>'
        b'<td class="race-runners__name"><a href="/dogs/501/dog-one" data-dog-id="501">DOG ONE</a></td></tr>'
        b'<tr class="race-runner"><td class="race-runners__finish-position">DNF</td>'
        b'<td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>'
        b'<td class="race-runners__name"><a href="/dogs/502/dog-two" data-dog-id="502">DOG TWO</a></td></tr></table>')
    attempt = tmp_path/'retention'/'attempts'/'exact-response'
    attempt.mkdir(parents=True, mode=0o700)
    body_path = attempt/'response.body'
    body_path.write_bytes(body)
    refs = {'body': runtime.reference(body_path)}
    refs['request'] = runtime.put_new(attempt/'request.json', {'at': captured.isoformat(), 'url': url})
    refs['response'] = runtime.put_new(attempt/'response.json', {'observed_at': captured.isoformat(),
        'final_url': url, 'status': 200, 'host': 'www.thedogs.com.au', 'retry_headers': {},
        'content_type': 'text/html', 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)})
    record = known_nonfinish_evidence(context['job'], context['bundle'], body, url,
        captured, f['clock']['now'], deadline=RetainedDeadline({'expires_at': '2030-01-01T00:00:00+00:00'}, None),
        prediction_bundles=context['prediction_bundles'], source_evidence=refs)
    directory = tmp_path/'retention'/'terminal-results'
    directory.mkdir(mode=0o700)
    runtime.put_new(directory/'job-fixture.json', record)
    f['database'].unlink()  # Non-finish is a separate native terminal, never fabricated full order.
    assert closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert report(f)['accounting']['scored_races'] == 1
    entry = json.loads((f['root']/'result-manifest.json').read_bytes())['entries'][0]
    assert entry['label_status'] == 'KNOWN_NONFINISH_WIN_ELIGIBLE'
    assert load(entry['target'])['outcome'] == [1., 0.]
    assert load(entry['closure_evidence'])['record'] == record


def test_native_runner_ids_cannot_be_reassigned_to_different_boxes(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    payload = load(runtime.reference(f['attempt']/'forecast.json'))
    payload['predictions'][0]['box_number'], payload['predictions'][1]['box_number'] = 2, 1
    with pytest.raises(closure.ClosureRejected, match='NATIVE_RESULT_CONTEXT_CHANGED'):
        closure._native_context(closure.evaluation._Checked(), payload)
