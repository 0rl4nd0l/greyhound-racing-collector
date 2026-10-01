"""Invented sealed jobs and official responses; no provider requests."""
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import fcntl
import json
import sqlite3
import subprocess

import pytest

from src.predictor.on_demand import canonical_bytes


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return ref(path)


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    from tests.fixtures.persistent_comparison_case import setup
    from src.operator_ui.job_store import JobStore
    from src.operator_ui.r3_api import build_verified_bundle_reader
    from src.predictor.comparison_result_runtime import database
    from scripts.autonomous_official_result_capture import ensure_official_result_evidence_tables
    import race_collection.persistent_storage as storage
    monkeypatch.setattr(storage, 'check_mount', lambda mount, path: None)
    setup(tmp_path)
    binding = json.loads((tmp_path/'binding.json').read_bytes())
    runtime = json.loads(Path(binding['authority']).read_bytes())['runtime']
    root = Path(runtime['state_root'])
    store = JobStore(Path(runtime['job_store']), readonly=True)
    job = store.recorded_jobs()[0]
    bundle = build_verified_bundle_reader(Path(runtime['prediction_bundles']), store)(job)
    captured = datetime.fromisoformat(job.input.jump_timestamp) + timedelta(minutes=20)
    attempt = root/'attempts/autonomous_official_result_capture_invented'
    attempt.mkdir(parents=True)
    rows = []
    for ordinal, runner in zip(('1st', '2nd', '3rd'), job.input.ordered_runners):
        rows.append(f'<tr class="race-runner"><td class="race-runners__finish-position">{ordinal}</td>'
                    f'<td class="race-runners__box"><sprite-svg name="rug_{runner["box"]}"></sprite-svg></td>'
                    f'<td class="race-runners__name"><a>{runner["name"]}</a>'
                    '<span class="race-runners__name__time">NBT</span></td></tr>')
    body = attempt/'response-invented.body'
    body.write_text('<table class="race-runners--result">'+''.join(rows)+'</table>')
    url = bundle.result['race']['url']+'?trial=false'
    request = put(attempt/'response-invented.request.json', {'at': captured.isoformat(), 'url': url})
    response = put(attempt/'response-invented.json', {'host': 'www.thedogs.com.au', 'status': 200,
        'observed_at': captured.isoformat(), 'retry_headers': {}, 'sha256': ref(body)['sha256'],
        'bytes': body.stat().st_size, 'final_url': url, 'content_type': 'text/html'})
    report = put(attempt/'official_result_ingest_dry_run_report.json', {'candidate_count': 1,
        'ingested': [], 'failed': [{'race_id': job.input.race_id,
            'errors': ['comparison_official_runner_identity_mismatch']} ]})
    with database(root) as db:
        db.execute('INSERT INTO identity VALUES(?)', (hashlib.sha256(canonical_bytes(binding)).hexdigest(),))
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)',
            (job.input.race_id, job.job_id, job.input.jump_timestamp, 'QUARANTINED', captured.isoformat(), 1))
        db.execute('INSERT INTO requests(at,race,artifact) VALUES(?,?,?)',
            (captured.isoformat(), job.input.race_id, str(body.with_suffix(''))))
        db.execute('INSERT INTO events(at,race,status,artifact) VALUES(?,?,?,?)',
            (captured.isoformat(), job.input.race_id, 'COLLECTOR_FAILURE', str(attempt)))
    with sqlite3.connect(root/'official-results.sqlite3') as db:
        ensure_official_result_evidence_tables(db)
    authority = {'schema_version': 'comparison_retained_identity_reconciliation_v1',
        'status': 'AUTHORIZED_RETAINED_IDENTITY_RECONCILIATION', 'authority_reference': 'SYNTHETIC_RECONCILIATION',
        'issued_at': captured.isoformat(), 'expires_at': (captured+timedelta(hours=1)).isoformat(),
        'binding': ref(tmp_path/'binding.json'), 'job_id': job.job_id, 'race_id': job.input.race_id,
        'attempt_directory': str(attempt), 'failed_report': report, 'request': request, 'response': response,
        'body': ref(body), 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'reconciliation_id': 'invented-identity-repair', 'network_requests_allowed': False,
        'outcomes_released': False, 'preserve_attempts': True}
    authority_path = tmp_path/'reconciliation-authority.json'
    put(authority_path, authority)
    # The production reconciler uses the actual clock in addition to an injected
    # admission time. Keep these invented historical fixtures calendar-independent.
    from scripts import reconcile_comparison_result_identity as reconciliation
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = captured+timedelta(minutes=1)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    monkeypatch.setattr(reconciliation, 'datetime', Clock)
    return dict(path=authority_path, authority=authority, runtime=runtime, root=root,
        now=captured+timedelta(minutes=1), job=job, bundle=bundle)


@pytest.mark.parametrize('with_date_header', [False, True])
def test_retained_identity_reconciliation_closes_without_consuming_or_refetching(case, with_date_header):
    from scripts.reconcile_comparison_result_identity import reconcile
    if with_date_header:
        path = Path(case['authority']['response']['path'])
        response = json.loads(path.read_bytes())
        response['retry_headers'] = {'date':'Thu, 01 Oct 2026 05:20:04 GMT'}
        case['authority']['response'] = put(path, response)
        put(case['path'], case['authority'])
    root = case['root']; original_report = Path(case['authority']['failed_report']['path']).read_bytes()
    ledger = Path(case['runtime']['campaign_root'])/'ledger.json'; before_ledger = ledger.read_bytes()
    result = reconcile(case['path'], ref(case['path'])['sha256'], 'SYNTHETIC_RECONCILIATION', now=case['now'])
    assert result['status'] == 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'
    assert result['provider_requests'] == 0 and result['outcomes_released'] is False
    assert ledger.read_bytes() == before_ledger
    assert Path(case['authority']['failed_report']['path']).read_bytes() == original_report
    with sqlite3.connect(root/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('CLOSED',1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert [r[0] for r in db.execute('SELECT status FROM events ORDER BY id')] == [
            'COLLECTOR_FAILURE','CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION']
    with pytest.raises((ValueError, FileExistsError)):
        reconcile(case['path'], ref(case['path'])['sha256'], 'SYNTHETIC_RECONCILIATION', now=case['now'])
    assert not Path(case['runtime']['lock_path']).exists()


def run(case):
    from scripts.reconcile_comparison_result_identity import reconcile
    put(case['path'], case['authority'])
    return reconcile(case['path'], ref(case['path'])['sha256'], 'SYNTHETIC_RECONCILIATION', now=case['now'])


@pytest.mark.parametrize('guidance', ['retry-after', 'ratelimit-reset', 'x-ratelimit-reset'])
def test_date_header_never_hides_provider_retry_guidance(case, guidance):
    path = Path(case['authority']['response']['path'])
    response = json.loads(path.read_bytes())
    response['retry_headers'] = {'date':'Thu, 01 Oct 2026 05:20:04 GMT', guidance:'600'}
    case['authority']['response'] = put(path, response)
    with pytest.raises(ValueError):
        run(case)
    assert_quarantined(case)


def assert_quarantined(case):
    with sqlite3.connect(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('QUARANTINED', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == 1
    assert not Path(case['runtime']['lock_path']).exists()


@pytest.mark.parametrize('mutation', ['name', 'missing', 'duplicate', 'bad_hash', 'bad_request', 'denied', 'wrong_failure'])
def test_rejects_nonexact_retained_evidence_without_append(case, mutation):
    authority = case['authority']
    body = Path(authority['body']['path'])
    response_path = Path(authority['response']['path'])
    response = json.loads(response_path.read_bytes())
    if mutation in ('name', 'missing', 'duplicate', 'bad_hash'):
        text = body.read_text()
        if mutation == 'name':
            text = text.replace(case['job'].input.ordered_runners[0]['name'], 'Different Invented Dog')
        elif mutation == 'missing':
            text = text[:text.index('<tr')] + text[text.index('</tr>')+5:]
        elif mutation == 'duplicate':
            text = text.replace('</table>', text[text.index('<tr'):text.index('</tr>')+5]+'</table>')
        else:
            text += '<!-- unexpected edit -->'
        body.write_text(text)
        if mutation != 'bad_hash':
            authority['body'] = ref(body)
            response.update(sha256=ref(body)['sha256'], bytes=body.stat().st_size)
            authority['response'] = put(response_path, response)
    elif mutation == 'bad_request':
        path = Path(authority['request']['path']); request = json.loads(path.read_bytes())
        request['at'] = (case['now']+timedelta(seconds=1)).isoformat()
        authority['request'] = put(path, request)
    elif mutation == 'denied':
        response.update(status=429, retry_headers={'retry-after':'600'})
        authority['response'] = put(response_path, response)
    else:
        path = Path(authority['failed_report']['path']); report = json.loads(path.read_bytes())
        report['failed'][0]['errors'] = ['some_other_failure']
        authority['failed_report'] = put(path, report)
    with pytest.raises(ValueError):
        run(case)
    assert_quarantined(case)
    with sqlite3.connect(case['root']/'official-results.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM autonomous_official_result_evidence_races').fetchone()[0] == 0


@pytest.mark.parametrize('guard', ['approval', 'expired', 'source', 'scope', 'membership', 'state', 'counter'])
def test_admission_guards_fail_before_protected_body_read(case, monkeypatch, guard):
    import scripts.reconcile_comparison_result_identity as module
    authority = case['authority']
    if guard == 'approval':
        authority['authority_reference'] = 'UNAPPROVED'
    elif guard == 'expired':
        authority['expires_at'] = case['now'].isoformat()
    elif guard == 'source':
        authority['source_commit'] = '0'*40
    elif guard == 'scope':
        binding_path = Path(authority['binding']['path']); binding = json.loads(binding_path.read_bytes())
        path = Path(binding['authority']); scope = json.loads(path.read_bytes())
        scope['human_outcome_access'] = True
        binding['authority_sha256'] = put(path, scope)['sha256']
        authority['binding'] = put(binding_path, binding)
    elif guard == 'membership':
        binding = json.loads(Path(authority['binding']['path']).read_bytes())
        plan = json.loads(Path(binding['plan']).read_bytes())
        key = hashlib.sha256(case['job'].input.race_id.encode()).hexdigest()
        (Path(plan['programme_root'])/binding['plan_sha256']/'attempts'/key/'completion.json').unlink()
    else:
        with sqlite3.connect(case['root']/'queue.sqlite3') as db:
            db.execute("UPDATE jobs SET state='PENDING'" if guard == 'state' else 'UPDATE jobs SET attempts=2')
    original = module.checked
    def checked(reference, **kwargs):
        assert reference != authority['body'], 'protected body reached before admission'
        return original(reference, **kwargs)
    monkeypatch.setattr(module, 'checked', checked)
    with pytest.raises(ValueError):
        run(case)
    assert not Path(case['runtime']['lock_path']).exists()


@pytest.mark.parametrize('lock', ['worker', 'owner', 'collector', 'queue'])
def test_live_locks_prevent_reconciliation(case, lock):
    from race_collection.synchronous_manual_capture import acquire_collector_lock_no_steal, release_owned_collector_lock
    if lock in ('worker', 'owner'):
        path = case['root']/'worker.lock' if lock == 'worker' else Path(case['runtime']['campaign_root'])/'owner.lock'
        with path.open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                run(case)
    elif lock == 'collector':
        owned = acquire_collector_lock_no_steal(Path(case['runtime']['lock_path']), run_id='other-owner',
            output_dir=case['root']/'other-output', phase='fixture')
        try:
            from race_collection.synchronous_manual_capture import CollectorBusy
            with pytest.raises(CollectorBusy):
                run(case)
            assert Path(case['runtime']['lock_path']).exists()
        finally:
            release_owned_collector_lock(owned)
    else:
        with sqlite3.connect(case['root']/'queue.sqlite3') as held:
            held.execute('BEGIN IMMEDIATE')
            with pytest.raises(sqlite3.OperationalError):
                run(case)
    assert_quarantined(case)


def test_interrupted_append_stays_quarantined_and_requires_new_explicit_id(case, monkeypatch):
    import scripts.autonomous_official_result_capture as native
    original = native.append_official_result_evidence_to_db
    def interrupted(**kwargs):
        original(**kwargs)
        raise RuntimeError('invented interruption after native append')
    monkeypatch.setattr(native, 'append_official_result_evidence_to_db', interrupted)
    with pytest.raises(RuntimeError):
        run(case)
    assert_quarantined(case)
    output = case['root']/'reconciliations'/case['authority']['reconciliation_id']
    assert (output/'before.json').is_file() and (output/'failure.json').is_file()
    assert not (output/'committed.json').exists()
    with sqlite3.connect(case['root']/'official-results.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM autonomous_official_result_evidence_races').fetchone()[0] == 1
    monkeypatch.setattr(native, 'append_official_result_evidence_to_db', original)
    with pytest.raises(FileExistsError):
        run(case)
    case['authority']['reconciliation_id'] = 'explicit-new-authorized-completion'
    result = run(case)
    assert result['status'] == 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'
    assert result['attempts_preserved'] == 1 and result['provider_requests'] == 0


def test_exported_entrypoint_denies_network_and_emits_only_structural_status(case, monkeypatch, capsys):
    import socket
    import scripts.reconcile_comparison_result_identity as module
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return case['now'].astimezone(tz) if tz else case['now'].replace(tzinfo=None)
    monkeypatch.setattr(module, 'datetime', Clock)
    monkeypatch.setattr('sys.argv', ['reconcile_comparison_result_identity', '--authority', str(case['path']),
        '--authority-sha256', ref(case['path'])['sha256'], '--approval-id', 'SYNTHETIC_RECONCILIATION'])
    assert module.main() == 0
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'
    assert set(value) == {'status', 'job_id', 'race_id', 'evidence_sha256', 'attempts_preserved',
                          'provider_requests', 'outcomes_released'}
    for family in (socket.AF_INET, socket.AF_INET6):
        with pytest.raises(PermissionError):
            socket.socket(family)


@pytest.mark.parametrize('status', ['SCR', 'L/SCR', 'LSCR'])
def test_explicit_extra_nonstarter_preserves_frozen_field_and_consumption(case, status):
    body = Path(case['authority']['body']['path'])
    extra = ('<tr class="race-runner"><td class="race-runners__finish-position">'+status+'</td>'
             '<td class="race-runners__box"><sprite-svg name="rug_8"></sprite-svg></td>'
             '<td class="race-runners__name"><a>Invented Nonstarter</a></td></tr>')
    body.write_text(body.read_text().replace('</table>', extra+'</table>'))
    case['authority']['body'] = ref(body)
    response_path = Path(case['authority']['response']['path'])
    response = json.loads(response_path.read_bytes())
    response.update(sha256=ref(body)['sha256'], bytes=body.stat().st_size)
    case['authority']['response'] = put(response_path, response)
    result = run(case)
    assert result['status'] == 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'
    assert result['provider_requests'] == 0
    with sqlite3.connect(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('CLOSED', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert [r[0] for r in db.execute('SELECT status FROM events ORDER BY id')] == [
            'COLLECTOR_FAILURE', 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION']


@pytest.mark.parametrize('status', ['DNF', 'FELL', 'DISQ', '', '4th'])
def test_extra_runner_without_explicit_nonstarter_proof_stays_quarantined(case, status):
    body = Path(case['authority']['body']['path'])
    extra = ('<tr class="race-runner"><td class="race-runners__finish-position">'+status+'</td>'
             '<td class="race-runners__box"><sprite-svg name="rug_8"></sprite-svg></td>'
             '<td class="race-runners__name"><a>Invented Extra Runner</a></td></tr>')
    body.write_text(body.read_text().replace('</table>', extra+'</table>'))
    case['authority']['body'] = ref(body)
    response_path = Path(case['authority']['response']['path'])
    response = json.loads(response_path.read_bytes())
    response.update(sha256=ref(body)['sha256'], bytes=body.stat().st_size)
    case['authority']['response'] = put(response_path, response)
    with pytest.raises(ValueError):
        run(case)
    assert_quarantined(case)
