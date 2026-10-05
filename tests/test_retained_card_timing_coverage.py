import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from race_collection import retained_card_timing_coverage as c

HEADER = ['Dog Name','Sex','PLC','BOX','WGT','DIST','DATE','TRACK','G','TIME','WIN','BON','1 SEC','MGN','W/2G','PIR','SP']


def card(rows=None):
    rows = rows or [
        {'Dog Name': '1. Alpha', 'DATE': '2026-09-28', 'TRACK':'GUNN', 'DIST':'340', 'TIME':'20.55', 'WIN':'20.11', 'BON':'19.9', '1 SEC':'6.5', 'PIR':'1234'},
        {'Dog Name': '', 'DATE': '2026-09-27', 'TRACK':'OTHER', 'DIST':'400', 'TIME':'21.5', 'WIN':'21.1', 'BON':'20', '1 SEC':'-', 'PIR':'-'},
        {'Dog Name': '2. Beta', 'DATE': '2026-09-28', 'TRACK':'GUNN', 'DIST':'340', 'TIME':'20.75', 'WIN':'20.1', 'BON':'19.9', '1 SEC':'6.7', 'PIR':'4321'},
    ]
    out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=HEADER,delimiter='|',lineterminator='\n');writer.writeheader()
    for row in rows:writer.writerow({**{'PLC':'SENSITIVE_PLACEMENT','SP':'SENSITIVE_ODDS'},**row})
    return out.getvalue().encode()


def sidecar():
    return {'runner_completeness_after_canonical_alignment':{'runner_count':2,'participants':[
        {'box_number':1,'dog_name':'Alpha'}, {'box_number':2,'dog_name':'Beta'}]}}


def member():
    return {'race_id':'Race 1 - GUNN - 2026-10-01','source_race_date':'2026-10-01', 'venue':'GUNN',
            'target_distance_raw':'340m','target_runner_slots':2,'runner_set_sha256':'a'*64}


def test_real_retained_header_native_block_parser_counts_without_values():
    result=c.audit_card(card(),sidecar(),member())
    assert result['target_runner_slots']==2
    assert [r['retained_prior_rows'] for r in result['runners']]==[2,1]
    assert result['runners'][0]['field_presence']['1 SEC']=={'positive_numeric':1,'missing':1}
    assert result['whole_field_same_raw_track_distance_at_least_one_positive']['1 SEC'] is True
    output=json.dumps(result)
    for secret in ('Alpha','Beta','SENSITIVE_PLACEMENT','SENSITIVE_ODDS','20.55','6.5','1234'):
        assert secret not in output


@pytest.mark.parametrize('value,status',[('', 'missing'),('-', 'missing'),('N/A','missing'),('0','nonpositive_numeric'),('-1','nonpositive_numeric'),('nan','non_finite'),('inf','non_finite'),('xx','non_numeric'),('6.7','positive_numeric')])
def test_presence_categories(value,status):
    assert c.number_presence(value)==status


def test_same_day_future_invalid_date_excluded_and_conflicts_not_averaged():
    rows=[{'Dog Name':'1. Alpha','DATE':'2026-10-01','TRACK':'GUNN','DIST':'340','1 SEC':'6.5'},
          {'Dog Name':'','DATE':'2026-10-02','TRACK':'GUNN','DIST':'340','1 SEC':'6.5'},
          {'Dog Name':'','DATE':'invalid','TRACK':'GUNN','DIST':'340','1 SEC':'6.5'},
          {'Dog Name':'2. Beta','DATE':'2026-09-28','TRACK':'GUNN','DIST':'340','1 SEC':'6.5'},
          {'Dog Name':'','DATE':'2026-09-28','TRACK':'GUNN','DIST':'340','1 SEC':'6.6'}]
    r=c.audit_card(card(rows),sidecar(),member())
    assert r['runners'][0]['row_dispositions']=={'SAME_DAY_OR_LATER_EXCLUDED':2,'INVALID_OR_MISSING_DATE':1}
    assert r['runners'][1]['row_dispositions']=={'AMBIGUOUS_EVENT_KEY_CONFLICT':2}
    assert not r['whole_field_at_least_one_positive']['1 SEC']


def test_exact_duplicates_counted_without_claiming_distinct_native_event_identity():
    row={'DATE':'2026-09-28','TRACK':'GUNN','DIST':'340','1 SEC':'6.5'}
    r=c.audit_card(card([{'Dog Name':'1. Alpha',**row},{'Dog Name':'',**row},{'Dog Name':'2. Beta',**row}]),sidecar(),member())
    assert r['runners'][0]['retained_prior_rows']==1
    assert r['runners'][0]['row_dispositions']=={'EXACT_WHITELIST_DUPLICATE':1}


def test_raw_track_alias_not_inferred_comparable():
    raw=card().replace(b'GUNN',b'GUNNEDAH')
    r=c.audit_card(raw,sidecar(),member())
    assert r['whole_field_at_least_one_positive']['1 SEC']
    assert not r['whole_field_same_raw_track_distance_at_least_one_positive']['1 SEC']


@pytest.mark.parametrize('change',['wrong_roster','duplicate_identity','missing_column','row_width'])
def test_card_integrity_rejects(change):
    raw=card();side=sidecar()
    if change=='wrong_roster':side['runner_completeness_after_canonical_alignment']['participants'][1]['dog_name']='Gamma'
    elif change=='duplicate_identity':side['runner_completeness_after_canonical_alignment']['participants'][1]['dog_name']='Alpha'
    elif change=='missing_column':raw=raw.replace(b'1 SEC',b'other')
    else:raw+=b'x|x\n'
    with pytest.raises(ValueError):c.audit_card(raw,side,member())


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(data,dict):data=json.dumps(data).encode()
    path.write_bytes(data)
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def fixed_manifest(tmp_path):
    members=[]; originals=[]
    for i in range(82):
        m={**member(),'race_id':f'Race {i+1} - GUNN - 2026-10-01','jump_at':'2026-10-01T20:00:00+10:00'}
        base=tmp_path/f'card{i}'
        m['accepted_csv']=put(base/'source/card.csv',card());m['raw_export']=put(base/'raw.csv',card())
        m['primary_page']=put(base/'page.html',b'<html>opaque pre-jump card</html>')
        receipt={'body_sha256':m['primary_page']['sha256'],'status_code':200,'race_discovery_key':m['race_id'],'requested_url':f'https://source/race/{i}','capture_timestamp':'2026-10-01T19:54:00+10:00'}
        m['primary_receipt']=put(base/'receipt.json',receipt)
        s={**sidecar(),'target_distance':'340m','content_sha256':m['accepted_csv']['sha256'],'raw_content_sha256':m['raw_export']['sha256'],'race_url':receipt['requested_url'],'primary_race_page_evidence':{'body_sha256':m['primary_page']['sha256'],'receipt_sha256':m['primary_receipt']['sha256']}}
        m['sidecar']=put(base/'source/card.csv.metadata.json',s)
        m['bundle_manifest']=put(base/'bundle_manifest.json',{'files':{'source/card.csv':m['accepted_csv'],'source/card.csv.metadata.json':m['sidecar']}})
        m['admission']=put(base/'admission.json',{'race':{'race_id':m['race_id'],'race_date':m['source_race_date'],'venue':m['venue']}})
        originals.append({**m,'original_admitted_at':'2026-10-01T19:55:00+10:00','original_published_complete_at':'2026-10-01T19:55:01+10:00','result_status':'QUARANTINED' if i<8 else 'UNKNOWN'})
        members.append(m)
    original=put(tmp_path/'membership.json',{'members':originals})
    data={'schema_version':'retained_speed_breadth_metadata_manifest_v1','selection':'ALL_82_FIXED_MEMBERS_IRRESPECTIVE_OF_RESULT_STATUS','membership':original,'members':members}
    return put(tmp_path/'manifest.json',data),data


def test_exact82_all_dispositions_quarantines_not_input_filter_and_private_output(tmp_path):
    ref,_=fixed_manifest(tmp_path)
    result=c.run(ref,tmp_path/'out')
    assert result['cards']==82 and result['target_runner_slots']==164
    report=json.loads((tmp_path/'out/coverage.json').read_text())
    assert len(report['records'])==82 and report['measurement_semantics']=='UNQUALIFIED'
    assert (tmp_path/'out/coverage.json').stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):c.run(ref,tmp_path/'out')


@pytest.mark.parametrize('change',['tamper','receipt_after_jump','denominator','source_hash','unsafe_link','read_limit','deadline'])
def test_shared_failure_no_successful_partial_output(tmp_path,monkeypatch,change):
    ref,data=fixed_manifest(tmp_path)
    if change=='tamper':Path(data['members'][-1]['accepted_csv']['path']).write_bytes(b'tamper')
    elif change=='receipt_after_jump':
        m=data['members'][-1];p=Path(m['primary_receipt']['path']);r=json.loads(p.read_text());r['capture_timestamp']='2026-10-01T20:01:00+10:00';m['primary_receipt']=put(p,r)
    elif change=='denominator':data['members'].pop()
    elif change=='source_hash':data['members'][-1]['primary_page']['sha256']='0'*64
    elif change=='unsafe_link':
        p=Path(data['members'][-1]['raw_export']['path']);p.rename(p.with_suffix('.original'));p.symlink_to(p.with_suffix('.original'))
    elif change=='read_limit':monkeypatch.setattr(c,'MAX_READS',5)
    else:monkeypatch.setattr(c,'MAX_SECONDS',-1)
    ref=put(Path(ref['path']),data)
    with pytest.raises(c.CoverageRejected):c.run(ref,tmp_path/'out')
    assert not (tmp_path/'out/coverage.json').exists()
    assert json.loads((tmp_path/'out/FAILED.json').read_text())['status']=='FAILED_NO_SUCCESSFUL_COVERAGE'


def test_cli_default_off_does_not_read_nonexistent_path(tmp_path):
    result=subprocess.run([sys.executable,'-B','-m','scripts.audit_retained_card_timing_coverage','--manifest',str(tmp_path/'absent')],capture_output=True,text=True)
    assert result.returncode==0
    assert json.loads(result.stdout)['status']=='DEFAULT_OFF'


def test_expiry_during_output_write_cannot_publish_complete(tmp_path,monkeypatch):
    ref,_=fixed_manifest(tmp_path)
    original=c.os.fsync
    def expire(fd):
        original(fd)
        monkeypatch.setattr(c,'MAX_SECONDS',-1)
    monkeypatch.setattr(c.os,'fsync',expire)
    with pytest.raises(c.CoverageRejected,match='WALL_LIMIT'):
        c.run(ref,tmp_path/'out')
    assert not (tmp_path/'out/coverage.json').exists()
    assert (tmp_path/'out/FAILED.json').exists()


def test_finite_output_limit_prevents_successful_index(tmp_path,monkeypatch):
    ref,_=fixed_manifest(tmp_path)
    monkeypatch.setattr(c,'MAX_OUTPUT',1)
    with pytest.raises(c.CoverageRejected,match='OUTPUT_LIMIT'):
        c.run(ref,tmp_path/'out')
    assert not (tmp_path/'out/coverage.json').exists()
