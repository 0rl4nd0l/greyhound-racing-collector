"""Fixed-deadline regression tests using invented evidence and no network."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.predictor import comparison_result_runtime as runtime
from tests.fixtures.persistent_comparison_case import setup


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Native frozen prediction, admission, runtime authority and SQLite stores.
    setup(tmp_path)
    binding_path = tmp_path / 'binding.json'
    binding = json.loads(binding_path.read_bytes())
    authority = json.loads(Path(binding['authority']).read_bytes())
    cfg = authority['runtime']
    deadline = datetime.fromisoformat(cfg['expires_at'])

    class Clock(datetime):
        value = deadline - timedelta(seconds=1)

        @classmethod
        def now(cls, tz=None):
            return cls.value.astimezone(tz) if tz else cls.value.replace(tzinfo=None)

    monkeypatch.setattr(runtime, 'datetime', Clock)
    scenario = json.loads((tmp_path / 'scenario.json').read_bytes())
    output = tmp_path / 'private/attempts/fixture-crossing'
    output.mkdir(parents=True)
    from src.operator_ui.job_store import JobStore
    job = JobStore(Path(cfg['job_store']), readonly=True).recorded_jobs()[0]
    with runtime.database(tmp_path / 'private') as db:
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)',
                   (scenario['race'], job.job_id, scenario['jump'], 'RUNNING',
                    (Clock.value - timedelta(minutes=30)).astimezone(timezone.utc).isoformat(), 0))
    return SimpleNamespace(root=tmp_path, binding_path=binding_path, binding=binding,
                           cfg=cfg, deadline=deadline, clock=Clock, scenario=scenario, output=output)


@pytest.mark.parametrize('offset,http_status', [(0, 200), (1, 200), (0, 429)])
def test_response_crossing_deadline_retains_opaque_bytes_without_decoding(case, offset, http_status):
    from race_collection.freshness_campaign import Campaign
    transport = runtime.Transport(case.binding, case.cfg, case.root / 'private',
                                  case.output, Campaign(case.cfg['campaign_root']))
    url = 'https://www.thedogs.com.au/fixture?trial=false'
    transport.allowed, transport.race = {url}, case.scenario['race']
    decoded = []

    class OpaqueBytes(bytes):
        def decode(self, *args, **kwargs):
            decoded.append(case.clock.value)
            return super().decode(*args, **kwargs)

    class Raw:
        def read(self, *args, **kwargs):
            case.clock.value = case.deadline + timedelta(seconds=offset)
            return OpaqueBytes(b'<html>Unpublished fixture</html>')

    response = SimpleNamespace(raw=Raw(), status_code=http_status, url=url,
                               headers={'Content-Type': 'text/html'}, close=lambda: None)
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    expected = 'SOURCE_HOLD' if http_status == 429 else 'DEADLINE_EXPIRED'
    with pytest.raises(ValueError, match='RESULT_' + expected):
        transport.get(session, url)
    assert decoded == []
    assert len(list(case.output.glob('*.body'))) == 1
    with runtime.database(case.root / 'private') as db:
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT attempts FROM jobs').fetchone()[0] == 1
    assert json.loads((case.output / 'transport-status.json').read_bytes())['status'] == expected
    if http_status == 429:
        assert json.loads((case.root / 'campaign/ledger.json').read_bytes())['source_holds']
        with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
            transport.check_deadline()
        assert json.loads((case.output / 'transport-status.json').read_bytes())['status'] == 'SOURCE_HOLD'


def test_queue_crossing_deadline_never_reads_results_and_seals_after_writer_idle(case, monkeypatch):
    from scripts import run_comparison_result_queue as queue
    from src.predictor.comparison_results import ComparisonResultSource
    monkeypatch.setattr(queue, 'datetime', case.clock)
    calls = []

    def elapsed():
        calls.append(1)
        if len(calls) > 1:
            case.clock.value = case.deadline
        return float(len(calls))

    monkeypatch.setattr(queue.time, 'monotonic', elapsed)
    original = ComparisonResultSource.read
    reads = []

    def observed_read(self, *args, **kwargs):
        reads.append(case.clock.value)
        assert case.clock.value < case.deadline, 'protected result read started after deadline'
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ComparisonResultSource, 'read', observed_read)
    with runtime.database(case.root / 'private') as db:
        db.execute('UPDATE jobs SET attempts=1')
        db.execute('INSERT INTO requests(at,race,artifact) VALUES(?,?,?)',
                   (case.clock.value.isoformat(), case.scenario['race'], 'retained-consumed-request'))
    result = queue.cycle(case.binding_path)
    assert result['status'] == 'CLOSURE_SEALED'
    assert reads == []
    assert result['counts'] == {'DEADLINE_UNRESOLVED': 1}
    with runtime.database(case.root / 'private') as db:
        assert db.execute('SELECT attempts FROM jobs').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
    receipt = (case.root / 'private/closure/closure.json').read_bytes()
    assert queue.cycle(case.binding_path)['status'] == 'CLOSURE_SEALED'
    assert (case.root / 'private/closure/closure.json').read_bytes() == receipt


def test_parser_does_not_access_response_text_when_transport_returns_at_deadline(case):
    from scripts.ingest_results_for_date import TheDogsResultFetcher
    accesses = []

    class Response:
        status_code = 200

        @property
        def text(self):
            accesses.append(case.clock.value)
            return '<html>Unpublished fixture</html>'

    def get(*args, **kwargs):
        case.clock.value = case.deadline
        return Response()

    transport = runtime.Transport(case.binding, case.cfg, case.root / 'private', case.output, None)
    token = runtime.ACTIVE.set(transport)
    try:
        candidate = SimpleNamespace(race_id=case.scenario['race'], thedogs_slug='gunnedah',
                                    canonical_thedogs_url='https://www.thedogs.com.au/fixture')
        result = TheDogsResultFetcher(None, http_session=SimpleNamespace(get=get)).fetch(candidate)
        assert result.positions_by_box == {}
        assert accesses == []
        assert json.loads((case.output / 'transport-status.json').read_bytes())['status'] == 'DEADLINE_EXPIRED'
    finally:
        runtime.ACTIVE.reset(token)


def test_append_crossing_deadline_rolls_back_uncommitted_result_rows(case, monkeypatch):
    import sqlite3
    from scripts import autonomous_official_result_capture as capture
    from tests.test_autonomous_official_result_capture import _official_artifact_rows
    path = case.root / 'private/official-results.sqlite3'
    with sqlite3.connect(path) as db:
        capture.ensure_official_result_evidence_tables(db)
    original = sqlite3.connect
    inserts = []

    def connect(*args, **kwargs):
        db = original(*args, **kwargs)

        def trace(sql):
            if sql.lstrip().upper().startswith('INSERT'):
                inserts.append(True)
                case.clock.value = case.deadline

        db.set_trace_callback(trace)
        return db

    monkeypatch.setattr(sqlite3, 'connect', connect)
    token = runtime.ACTIVE.set(runtime.Transport(case.binding, case.cfg, case.root / 'private', case.output, None))
    try:
        with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
            capture.append_official_result_evidence_to_db(db_path=path, artifact_rows=_official_artifact_rows(),
                output_dir=case.output, execute=True)
    finally:
        runtime.ACTIVE.reset(token)
    assert inserts
    with original(path) as db:
        assert db.execute(f'SELECT count(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}').fetchone()[0] == 0
        assert db.execute(f'SELECT count(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}').fetchone()[0] == 0


@pytest.mark.parametrize('writer_lock', ['owner', 'collector'])
def test_child_crossing_deadline_keeps_consumption_and_waits_for_writer_before_closure(case, monkeypatch, writer_lock):
    import fcntl
    import subprocess
    from scripts import run_comparison_result_queue as queue
    from src.predictor.comparison_results import ComparisonResultSource
    monkeypatch.setattr(queue, 'datetime', case.clock)
    reads = []
    original_read = ComparisonResultSource.read

    def read(self, *args, **kwargs):
        assert case.clock.value < case.deadline
        reads.append(case.clock.value)
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(ComparisonResultSource, 'read', read)
    owner = (case.root / 'campaign/owner.lock').open('a')
    original_popen = subprocess.Popen

    class Child:
        def wait(self, **kwargs):
            # The subprocess boundary alone is substituted. Attempt accounting
            # is durably charged as in Transport.get; the native parent resumes.
            with runtime.database(case.root / 'private') as db:
                db.execute('UPDATE jobs SET attempts=attempts+1')
                db.execute('INSERT INTO requests(at,race,artifact) VALUES(?,?,?)',
                           (case.clock.value.isoformat(), case.scenario['race'], 'fixture-child'))
            if writer_lock == 'owner':
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                (case.root / 'collector.lock').write_text('fixture writer still owns lock')
            case.clock.value = case.deadline
            return 0

    def popen(command, **kwargs):
        return Child() if command[2:4] == ['-m', 'scripts.autonomous_official_result_capture'] else original_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    try:
        result = queue.cycle(case.binding_path)
        assert result['status'] == 'CLOSURE_WRITER_BUSY'
        assert result['counts'] == {'RUNNING': 1}
        assert not (case.root / 'private/closure').exists()
    finally:
        owner.close()
        if writer_lock == 'collector':
            (case.root / 'collector.lock').unlink()
    assert len(reads) == 1
    assert queue.cycle(case.binding_path)['status'] == 'CLOSURE_SEALED'
    with runtime.database(case.root / 'private') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone()[:] == ('DEADLINE_UNRESOLVED', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1


def test_direct_collector_rejects_exact_cutoff_before_entering_body(case):
    case.clock.value = case.deadline
    with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
        with runtime.collector_guard(case.binding, output=case.output,
                job_store=Path(case.cfg['job_store']), bundles=Path(case.cfg['prediction_bundles']),
                result_database=case.root / 'private/official-results.sqlite3'):
            pytest.fail('expired collector entered body')
    with runtime.database(case.root / 'private') as db:
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 0
    assert not (case.root / 'collector.lock').exists()


def test_queue_startup_crossing_deadline_does_not_open_job_store(case, monkeypatch):
    import sqlite3
    from scripts import run_comparison_result_queue as queue
    monkeypatch.setattr(queue, 'datetime', case.clock)
    original_exists, original_connect = Path.exists, sqlite3.connect
    opened = []

    def exists(path):
        value = original_exists(path)
        if path == Path(case.cfg['job_store']):
            case.clock.value = case.deadline
        return value

    def connect(path, *args, **kwargs):
        if Path(case.cfg['job_store']).name in str(path):
            opened.append(str(path))
            assert case.clock.value < case.deadline, 'expired startup opened job store'
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'exists', exists)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    assert queue.cycle(case.binding_path)['status'] == 'CLOSURE_SEALED'
    assert opened == []
