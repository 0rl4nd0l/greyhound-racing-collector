from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from race_collection import scheduled_input_retention as scheduled
from race_collection.retained_feature_replay import replay_retained_inputs
from tests.test_prospective_input_retention import generator_files


def test_unapproved_config_never_stats_or_opens_history(tmp_path, monkeypatch):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'schema_version':'scheduled_input_retention_v1','authority':{'approved':False}}))
    operation = scheduled.ScheduledInputRetention(config_path=config, evidence_root=tmp_path,
        protocol_root=tmp_path, collector_run_id='fixture', history_source=tmp_path/'DO_NOT_OPEN.db')
    monkeypatch.setattr(scheduled.subprocess, 'Popen', lambda *a,**k: pytest.fail('worker must not start'))
    assert operation(plan_item={'race_id':'fixture'}, attempt={}, receipt_publish={}) == {
        'status':'REJECTED','reason':'HISTORY_ACCESS_NOT_AUTHORIZED'}


def test_time_budget_kills_worker_group_and_preserves_consumed_failure(tmp_path, monkeypatch):
    import subprocess
    now = scheduled._now()
    source = tmp_path/'synthetic.db'
    source.write_bytes(b'fixture stat only; never parsed')
    config = {'schema_version':'scheduled_input_retention_v1',
        'authority':{'approved':True,'scope':scheduled.SCOPE,'approval_reference':'synthetic-fixture',
            'race_ids':['fixture'],'history_source':str(source),
            'not_before':(now-timedelta(seconds=10)).isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()},
        'max_seconds':0.25,'cutoff_seconds_before_jump':60,'max_history_source_bytes':1000,
        'max_bundle_bytes':1000,'output_root':str(tmp_path/'retention')}
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    operation = scheduled.ScheduledInputRetention(config_path=config_path, evidence_root=tmp_path,
        protocol_root=tmp_path, collector_run_id='fixture', history_source=source)
    calls = []
    class Worker:
        pid = 999999999
        def wait(self, timeout=None):
            calls.append(timeout)
            if timeout is not None: raise subprocess.TimeoutExpired('fixture', timeout)
    monkeypatch.setattr(scheduled.subprocess, 'Popen', lambda *a,**kw: Worker())
    monkeypatch.setattr(scheduled.os, 'killpg', lambda pid,sig: calls.append('killed-group'))
    result = operation(plan_item={'race_id':'fixture','jump_datetime':(now+timedelta(minutes=20)).isoformat()}, attempt={}, receipt_publish={})
    assert result['reason'] == 'RETENTION_TIME_BUDGET_EXCEEDED'
    assert calls == [0.25, 'killed-group', None]
    assert len(list((tmp_path/'retention').glob('*/terminal.json'))) == 1
    assert not list((tmp_path/'retention').glob('*/bundle'))
    assert operation(plan_item={'race_id':'fixture','jump_datetime':(now+timedelta(minutes=20)).isoformat()}, attempt={}, receipt_publish={})['reason'] == 'RUN_RETENTION_BUDGET_EXHAUSTED'


def test_actual_scheduled_receipt_to_retention_and_replay(tmp_path, monkeypatch):
    # Reuse the existing genuine receipt producer/verifier fixture. No stubs
    # replace receipt authentication, history sealing or feature generation.
    fixture_file = Path(__file__).parent/'race_collection/test_scheduled_forward_corpus.py'
    spec = importlib.util.spec_from_file_location('retention_scheduled_fixture', fixture_file)
    fixture_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture_module
    spec.loader.exec_module(fixture_module)
    from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence
    def add_page(sidecar):
        page = TimedResponse(requested_url=fixture_module.RACE_URL, final_url=fixture_module.RACE_URL,
            request_start_utc=fixture_module.NOW-timedelta(seconds=1), request_end_utc=fixture_module.NOW,
            status_code=200, headers={}, body=b'<html>invented pre-race source</html>')
        sidecar['primary_race_page_evidence'] = persist_primary_race_page_evidence(
            artifact_root=Path(sidecar['raw_export_path']).parent.parent,
            race_discovery_key=fixture_module.RACE_ID, response=page, canonical_runner_set={})
    case = fixture_module._fixture(tmp_path, mutate_sidecar=add_page)
    db = tmp_path/'source.db'
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE race_metadata(race_id TEXT,race_date TEXT,data_source TEXT,url TEXT); CREATE TABLE dog_race_data(race_id TEXT,dog_name TEXT,finish_position INTEGER,data_source TEXT); INSERT INTO race_metadata VALUES ('prior','2026-08-01','fixture','https://fixture.test/prior'); INSERT INTO dog_race_data VALUES ('prior','Alpha',1,'fixture'),('prior','Bravo',2,'fixture');")
    static = {k:{'path':str(p),'sha256':h} for k,(p,h) in generator_files(tmp_path).items()}
    claim = tmp_path/'claim'
    claim.mkdir()
    request = {'config':{'static_files':static,'max_bundle_bytes':10_000_000},
        'config_sha256':'a'*64,
        'context':{'protocol_root':str(case.protocol.root),'evidence_root':str(case.evidence_root),
                   'collector_run_id':'fixture-scheduled-collector','history_source':str(db)},
        'plan_item':case.plan_item,'attempt':case.attempt,'receipt_publish':case.receipt_publish,
        'cutoff':(fixture_module.JUMP-timedelta(minutes=1)).isoformat()}
    # Obtain the actual producer's collector run identity from its safe wrapper.
    receipt = next((case.protocol.root/'collector-exact-receipts').rglob('*.json'))
    request['context']['collector_run_id'] = json.loads(receipt.read_bytes())['collector_run_id']
    path = claim/'request.json'
    path.write_text(json.dumps(request))
    monkeypatch.setattr(scheduled, '_now', lambda: fixture_module.NOW+timedelta(seconds=5))
    result = scheduled._retain(path)
    assert result['status'] == 'RETAINED'
    bundle = claim/'bundle'
    # A completed worker is not sufficient: parent acceptance must be bound.
    with pytest.raises(FileNotFoundError): replay_retained_inputs(bundle)
    for status, digest in [('REJECTED', result['manifest_sha256']), ('RETAINED', '0'*64)]:
        (claim/'terminal.json').write_text(json.dumps({**result,'status':status,'manifest_sha256':digest,
            'config_sha256':'a'*64,'accepted_at':(fixture_module.NOW+timedelta(seconds=6)).isoformat()}))
        with pytest.raises(ValueError, match='SCHEDULED_RETENTION_NOT_ACCEPTED'): replay_retained_inputs(bundle)
    (claim/'terminal.json').write_text(json.dumps({**result,'config_sha256':'a'*64,
        'accepted_at':(fixture_module.NOW+timedelta(seconds=6)).isoformat()}))
    db.unlink()
    assert replay_retained_inputs(bundle)['status'] == 'IDENTICAL_FEATURE_REPLAY'
