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
    assert saved['status'] == 'INPUTS_RETAINED_NOT_QUALIFIED'
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
