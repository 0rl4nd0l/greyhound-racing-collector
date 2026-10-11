import importlib.util
import json
import sqlite3
from pathlib import Path

spec = importlib.util.spec_from_file_location('project_results', Path(__file__).parents[1]/'scripts/project_live_benchmark_results.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_only_authorised_exact_race_rows_are_projected(tmp_path):
    db = tmp_path/'results.db'
    c = sqlite3.connect(db)
    race_columns = 'race_id,race_date,venue,race_number,source,source_url,status,position_count,participant_count,participant_source,captured_at,source_artifact_dir,row_json'
    runner_columns = 'race_id,source,source_url,box_number,dog_name,finish_position,is_winner,captured_at,source_artifact_dir,row_json'
    c.execute('create table autonomous_official_result_evidence_races ('+race_columns+')')
    c.execute('create table autonomous_official_result_evidence_runners ('+runner_columns+')')
    for race in ('allowed', 'reserved', 'allowed-suffix'):
        c.execute('insert into autonomous_official_result_evidence_races values ('+','.join('?'*13)+')', [race]+['secret-result' if race != 'allowed' else 'safe']*12)
        c.execute('insert into autonomous_official_result_evidence_runners values ('+','.join('?'*10)+')', [race]+['secret-result' if race != 'allowed' else 'safe']*9)
    c.commit()
    c.close()
    membership = tmp_path/'membership.json'
    membership.write_text(json.dumps({'records': [
        {'race_id': 'allowed', 'access_disposition': 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL'},
        {'race_id': 'reserved', 'access_disposition': 'EXCLUDED_SCIENTIFIC_ALLOCATION'}]}))
    before = db.read_bytes()
    result = m.project(membership, db)
    assert result['result_races'] == 1
    assert 'secret-result' not in json.dumps(result)
    assert db.read_bytes() == before
