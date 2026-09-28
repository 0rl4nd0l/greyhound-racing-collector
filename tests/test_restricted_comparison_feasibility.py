from datetime import date
from pathlib import Path
import json,sqlite3
import pytest
from src.predictor.research_input_firewall import InputBoundary,json_keys,project_metadata,card_dates_before_decode,database_dates_before_decode


def test_json_projection_rejects_target_outcome_before_value_decoding():
    raw=b'{"race_info":{"date":"2026-09-28"},"results":{"winner":"\xff"}}'
    assert 'winner' in json_keys(raw)
    assert project_metadata(raw,['race_info.date'])=={'race_info':{'date':'2026-09-28'}}


@pytest.mark.parametrize('day', ['2026-09-28','2026-09-29',''])
def test_card_result_cell_never_decoded_on_unsafe_date(day):
    raw=b'Dog Name|PLC|DATE\n1. Invented|\xff|'+day.encode()+b'\n'
    with pytest.raises(InputBoundary):card_dates_before_decode(raw,date(2026,9,28),date(2026,9,28))


def test_quoted_framing_and_strictly_earlier_permission():
    raw=b'Dog Name,PLC,DATE\n"1. Invented, Dog",1,2026-09-27\n'
    assert card_dates_before_decode(raw,date(2026,9,28),date(2026,9,28))==[date(2026,9,27)]


@pytest.mark.parametrize('kind',['target','future','orphan','earlier'])
def test_database_projection_reads_no_outcome_column(tmp_path,kind):
    db=tmp_path/'history.db'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE race_metadata(race_id TEXT,race_date TEXT); CREATE TABLE dog_race_data(race_id TEXT,finish_position BLOB);')
        rid='target' if kind=='target' else 'prior';day='2026-09-29' if kind=='future' else '2026-09-27'
        conn.execute('INSERT INTO race_metadata VALUES (?,?)',(rid,day));conn.execute('INSERT INTO dog_race_data VALUES (?,?)',('missing' if kind=='orphan' else rid,b'\xff'))
    if kind=='earlier':assert database_dates_before_decode(db,'target',date(2026,9,28),date(2026,9,28))==[date(2026,9,27)]
    else:
        with pytest.raises(InputBoundary):database_dates_before_decode(db,'target',date(2026,9,28),date(2026,9,28))


def test_failed_worker_egress_never_contains_source_exception(tmp_path,monkeypatch):
    from scripts.restricted_comparison_feasibility import assess
    import scripts.restricted_comparison_feasibility as module
    def fail(*a):raise RuntimeError('SECRET_TARGET_WINNER_FEATURE_VALUE')
    monkeypatch.setattr(module,'load_pinned',fail)
    result=assess({'sample_id':'a'*64,'venue':'synthetic','date':'2026-09-28','field_size':6,'retained_manifest_path':str(tmp_path/'manifest.json'),'retained_manifest_sha256':'b'*64})
    assert 'SECRET' not in json.dumps(result) and result['reason']=='EXECUTION_FAILED_RuntimeError'


def test_unknown_and_embedded_result_values_are_never_decoded():
    raw=b'{"captured_at":"2026-09-28","finish":"\xff","unknown":{"position":"\xff"},"embedded":"{\\\"winner\\\":1}"}'
    assert project_metadata(raw,['captured_at'])=={'captured_at':'2026-09-28'}


def test_undated_unknown_csv_result_column_rejected_before_decode():
    with pytest.raises(InputBoundary,match='UNDATED_NON_ROSTER'):
        card_dates_before_decode(b'Dog Name|PLC|DATE|winnerName\n1. Dog|||\xff\n',date(2026,9,28),date(2026,9,28))


def test_unapproved_retained_worker_rejected():
    from scripts.restricted_comparison_feasibility import verify_replay_source
    with pytest.raises(InputBoundary,match='UNAPPROVED_REPLAY_WORKER'):
        verify_replay_source({'feature_replay_worker':b'print("not approved")'})
