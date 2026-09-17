from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import pytest

from race_collection.prospective_input_retention import (
    REQUIRED_ROLES, RetentionRejected, retain_inputs,
)

NOW = datetime(2030, 1, 2, 1, tzinfo=timezone.utc)


def fixture(tmp_path):
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE race_metadata(race_id TEXT, race_date TEXT);
            CREATE TABLE dog_race_data(race_id TEXT, dog_name TEXT, finish_position INTEGER);
            INSERT INTO race_metadata VALUES ('earlier', '2030-01-01'), ('target', '2030-01-02'), ('other_same_day', '2030-01-02'), ('future', '2030-01-03');
            INSERT INTO dog_race_data VALUES ('earlier', 'Invented Dog', 2), ('target', 'Invented Dog', 1), ('other_same_day', 'Invented Dog', 4), ('future', 'Invented Dog', 3);
        """)
    files = {}
    for role in REQUIRED_ROLES:
        path = tmp_path / role
        raw = ('synthetic-' + role).encode()
        path.write_bytes(raw)
        files[role] = (path, hashlib.sha256(raw).hexdigest())
    return dict(destination=tmp_path / 'retained', race_id='target',
                runner_names=['Invented Dog'], observed_at=NOW-timedelta(seconds=1),
                prediction_cutoff=NOW+timedelta(minutes=2), jump_at=NOW+timedelta(minutes=5),
                history_source=db, files=files, clock=lambda: NOW)


def test_complete_input_archive_survives_source_change_and_excludes_target(tmp_path):
    args = fixture(tmp_path)
    manifest = retain_inputs(**args)
    args['files']['normalized_form'][0].write_bytes(b'changed-later')
    saved = json.loads((args['destination']/'manifest.json').read_text())
    assert saved == manifest
    for entry in saved['files'].values():
        assert hashlib.sha256((args['destination']/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    with sqlite3.connect(args['destination']/'history.db') as conn:
        assert conn.execute('SELECT race_id FROM dog_race_data').fetchall() == [('earlier',)]
    assert saved['status'] == 'INPUTS_PENDING_COMPLETION'
    assert json.loads((args['destination']/'completion.json').read_text())['status'] == 'INPUTS_RETAINED_NOT_QUALIFIED'
    assert saved['predictions_generated'] is False
    with pytest.raises(RetentionRejected, match='DESTINATION_EXISTS'):
        retain_inputs(**args)


@pytest.mark.parametrize('failure', ['missing', 'changed', 'late_start', 'late_finish'])
def test_incomplete_or_late_inputs_never_publish_complete_manifest(tmp_path, failure):
    args = fixture(tmp_path)
    if failure == 'missing': args['files'].pop('raw_form')
    if failure == 'changed': args['files']['model'][0].write_bytes(b'drift')
    if failure == 'late_start': args['clock'] = lambda: args['prediction_cutoff']
    if failure == 'late_finish':
        times = iter([NOW, args['prediction_cutoff']])
        args['clock'] = lambda: next(times)
    with pytest.raises(RetentionRejected): retain_inputs(**args)
    assert not args['destination'].exists()


def test_cli_rejects_inventory_drift_before_processing(tmp_path, monkeypatch, capsys):
    from scripts import retain_prospective_inputs as cli
    inventory = tmp_path/'inventory.json'
    inventory.write_text('{"private": "must never be logged"}')
    monkeypatch.setattr('sys.argv', ['retain', '--inventory', str(inventory),
        '--inventory-sha256', '0'*64, '--destination', str(tmp_path/'output')])
    monkeypatch.setattr(cli, 'retain_inputs', lambda **kw: pytest.fail('must not process'))
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        'status': 'INPUT_RETENTION_FAILED', 'reason': 'INVENTORY_HASH_MISMATCH',
        'predictions_generated': False,
    }


def generator_files(tmp_path):
    """Actual generator source and model, synthetic input data only."""
    import importlib.metadata
    import platform
    import zipfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path/'generator.zip'
    paths = [root/'scripts/__init__.py', root/'scripts/utils.py', root/'scripts/run_shadow_non_tgr_rf_evaluation.py',
             root/'scripts/run_feature_recovery_execution_v1.py']
    paths += list((root/'utils').glob('*.py')) + list((root/'config').glob('*.py'))
    with zipfile.ZipFile(archive, 'w') as z:
        for path in paths: z.write(path, path.relative_to(root))
    lock = tmp_path/'environment.json'
    lock.write_text(json.dumps({'python':platform.python_version(), 'packages':{
        name:importlib.metadata.version(name) for name in ['requests','urllib3','certifi','charset-normalizer','idna']}}))
    sources = {'model':root/'artifacts/frozen_models/market_form_residual_v1/model.json',
               'model_manifest':root/'artifacts/frozen_models/market_form_residual_v1/manifest.json',
               'configuration':root/'configs/prediction/manual-default.json',
               'feature_schema':root/'accuracy_program/repaired_non_tgr_schema.json',
               'feature_replay_worker':root/'scripts/retained_feature_worker.py',
               'generator_source_archive':archive, 'environment_lock':lock}
    return {k:(v,hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in sources.items()}


def test_real_generator_replays_after_original_inputs_are_removed(tmp_path):
    from race_collection.retained_feature_replay import replay_retained_inputs
    args = fixture(tmp_path)
    form = tmp_path/'Race 1 - TRA - 2030-01-02.csv'
    form.write_text('Dog Name|BOX|DATE|TRACK|DIST|G|PLC|TIME|MGN\n1. Invented Dog|1|2029-12-30|TRA|350|Grade 5|1|19.0|0.0\n2. No History|2|||||||\n')
    args['runner_names'].append('No History')
    sidecar = tmp_path/(form.name+'.metadata.json')
    sidecar.write_text(json.dumps({'metadata_is_leakage_safe':True,
        'metadata_source_url':'https://www.thedogs.com.au/racing/traralgon/2030-01-02/1/test',
        'target_distance':'350m','target_distance_source':'canonical_pre_race_page',
        'target_grade':'Grade 5','target_grade_source':'canonical_pre_race_page',
        'race_info':{'date':'2030-01-02','venue':'TRA','race_number':1,'race_time':'12:05'}}))
    args['files'].update(generator_files(tmp_path))
    args['files'].update({k:(v,hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in {'normalized_form':form,'form_metadata':sidecar}.items()})
    with sqlite3.connect(args['history_source']) as conn:
        conn.execute("INSERT INTO dog_race_data VALUES ('earlier', 'Unrelated Dog', 1)")
    from scripts.run_shadow_non_tgr_rf_evaluation import build_live_feature_rows
    original = build_live_feature_rows(input_paths=[form], schema=json.loads(args['files']['feature_schema'][0].read_bytes()), db_path=args['history_source'])
    manifest = retain_inputs(**args, generate_features=True)
    saved = json.loads((args['destination']/'feature_values.json').read_bytes())
    for before, after in zip(original, saved):
        assert {name: before[name] for name in after['features']} == after['features']
    assert saved[1]['features']['days_since_last_start'] is None
    assert len(saved[0]['features']) == 16
    assert saved[0]['features']['prior_start_count'] == 2  # DB plus embedded form
    assert saved[0]['features']['recent_finish_mean_3'] == 1.5
    assert saved[0]['features']['recent_avg_margin_5'] == 0.0
    assert saved[0]['features']['same_grade_start_count'] == 1
    with sqlite3.connect(args['destination']/'history.db') as conn:
        assert conn.execute('SELECT dog_name FROM dog_race_data').fetchall() == [('Invented Dog',)]
    args['history_source'].unlink()
    form.unlink()
    sidecar.unlink()
    # Replay is a fresh isolated interpreter importing the retained ZIP, with
    # all non-retained data reads, writes, sockets and subprocesses denied.
    assert replay_retained_inputs(args['destination'])['feature_values_sha256'] == manifest['feature_values_sha256']


def test_completion_not_published_if_manifest_flush_crosses_cutoff(tmp_path):
    args = fixture(tmp_path)
    times = iter([NOW, NOW, NOW, args['prediction_cutoff']])
    args['clock'] = lambda: next(times)
    with pytest.raises(RetentionRejected, match='CUTOFF_PASSED'):
        retain_inputs(**args)
    assert not args['destination'].exists()


def test_safe_history_failure_code_survives_without_private_details():
    from race_collection.prospective_input_retention import failure_code
    from src.predictor.on_demand import PredictionBlocked
    assert failure_code(PredictionBlocked('HISTORY_DATABASE_BUSY', private='never display')) == 'HISTORY_DATABASE_BUSY'
