"""Fabricated four-surface fixtures retain source shape, never real outcomes."""
import hashlib
import json
from pathlib import Path

import pytest

from race_collection.retained_speed_history import audit_manifest
from scripts.audit_retained_speed_history import main

URL = 'https://www.thedogs.com.au/racing/example/2026-10-04/8/test?trial=false'
RUNNER = 'https://www.thedogs.com.au/dogs/runner/123'
HEADERS = ['DATE', 'TRACK', 'DIST', 'TIME', 'WIN', 'BON', '1 SEC', 'PIR']


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def history(expert=False, *, duplicate=False, conflict=False, future=False, unknown=False):
    classname = 'race-runners--expert' if expert else 'runner-form'
    rows = []
    for n in range(5):
        epoch = 1800000000 if future and n == 0 else 1700000000 + 86400 * n
        date = f'<a href="/racing/event/{n}"><span data-timestamp="{epoch}">date</span></a>'
        cells = [date, 'XYZ' if expert else 'XYZLONG', '400', '21.99', '21.00', '20.90', '-',
                 '<span class="runner-form__in-running-places">fictional-code</span>']
        if unknown and n == 0:cells[0] = 'unknown'
        row = '<tr>' + ''.join('<td>' + c + '</td>' for c in cells) + '</tr>'
        rows.append(row)
    if duplicate:
        row = rows[-1].replace('<tr>', '<tr class="runner-form__last-win">')
        if conflict:row = row.replace('21.99', '22.00')
        rows.append(row)
    profile = '' if expert else '<div data-dog-id="999"></div>'
    return profile + f'<table class="{classname}"><thead><tr>' + ''.join(
        '<th>' + h + '</th>' for h in HEADERS) + '</tr></thead><tbody>' + ''.join(rows) + \
        '</tbody></table><button data-runner-show-more="true"></button>'


@pytest.fixture
def scope(tmp_path):
    standard = '<table><tbody data-content-url="/dogs/runner/123"><tr data-runner-id="123"><td data-dog-id="999">runner</td></tr></tbody></table>'
    expert = '<div class="layout--sidebar--expert"><i data-runner-id="123" data-dog-id="999"></i></div><table-loader data-src="/dogs/runner/123/expert-form"></table-loader><select>' + ''.join(
        f'<option value="{key}">{label}</option>' for label, key in [('TIME','finish_time'),
        ('WIN','race_finish_time'),('BON','best_of_night_time'),('1 SEC','first_sectional_time')]) + '</select>'
    surfaces = {}
    for role, body, url in [('standard', standard, URL), ('expert', expert, URL.split('?')[0] + '/expert-form'),
                            ('runner', history(duplicate=True), RUNNER), ('runner_expert', history(True), RUNNER + '/expert-form')]:
        p = tmp_path/(role + '.html');p.write_text(body);b = ref(p)
        if role == 'standard':
            receipt = dict(schema_version='thedogs_primary_race_page_evidence_v1',requested_url=url,final_url=url,
                status_code=200,body_sha256=b['sha256'],content_length=len(p.read_bytes()),
                request_start_utc='2026-10-04T09:00:00Z',request_end_utc='2026-10-04T09:01:00Z',
                headers={'content-type':'text/html'})
        else:
            receipt = dict(kind=role,url=url,final_url=url,http_status=200,method='GET',redirects=False,retries=0,
                body=b,body_bytes=len(p.read_bytes()),started_at='2026-10-04T09:02:00Z' if role=='expert' else '2026-10-04T09:04:00Z',
                ended_at='2026-10-04T09:03:00Z' if role=='expert' else '2026-10-04T09:05:00Z',
                content_type='text/html',retry_headers={},status='RETAINED_HTTP_200')
        rp=tmp_path/(role+'.receipt.json');rp.write_text(json.dumps(receipt))
        surfaces[role]=dict(url=url,receipt=ref(rp),body=b)
    manifest=dict(schema_version='retained_speed_history_scope_v1',max_cards=1,max_runners=1,
        target_jump='2026-10-04T20:42:00+11:00',surfaces=surfaces)
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    return p


def alter(scope, role, *, body=None, receipt=None, item=None):
    manifest=json.loads(scope.read_text());entry=manifest['surfaces'][role]
    rp=Path(entry['receipt']['path']);r=json.loads(rp.read_text())
    if body:
        p=Path(entry['body']['path']);p.write_text(body(p.read_text()));entry['body']=ref(p)
        if role=='standard':r.update(body_sha256=entry['body']['sha256'],content_length=p.stat().st_size)
        else:r.update(body=entry['body'],body_bytes=p.stat().st_size)
    if receipt:receipt(r)
    rp.write_text(json.dumps(r));entry['receipt']=ref(rp)
    if item:item(entry)
    scope.write_text(json.dumps(manifest))


def test_five_events_not_six_starts_and_no_raw_values(scope):
    out=audit_manifest(ref(scope));normal=out['normal']
    assert normal['rendered_rows']==6 and normal['unique_identity_rows']==5
    assert normal['deduplicated_last_win_rows']==1 and out['shared_event_observations']==5
    assert out['equal_raw_track_labels']==0 and out['equal_raw_distance_labels']==5
    assert normal['field_presence']['1 SEC']==dict(present=0,numeric=0,missing=5)
    assert out['usable_early_speed_events']==0 and normal['show_more_present']
    assert set(out['source_label_definitions'].values())=={'SOURCE_SORT_KEY_CONFIRMED'}
    encoded=json.dumps(out)
    for raw in ['21.99','21.00','20.90','fictional-code','XYZLONG','1700000000']:
        assert raw not in encoded


@pytest.mark.parametrize('defect', ['hash','body_limit','denial','wrong_url','after_jump','naive_time',
    'wrong_profile','wrong_entry','wrong_lazy','detail_identity','retry','wrong_parent'])
def test_binding_failures_reject(scope,defect):
    if defect=='hash':
        m=json.loads(scope.read_text());Path(m['surfaces']['runner']['body']['path']).write_text('changed')
    elif defect=='body_limit':alter(scope,'runner',body=lambda _: 'x'*(4*1024*1024+1))
    elif defect=='denial':alter(scope,'runner',receipt=lambda r:r.update(http_status=403))
    elif defect=='wrong_url':alter(scope,'runner',receipt=lambda r:r.update(final_url=RUNNER+'0'))
    elif defect=='after_jump':alter(scope,'runner',receipt=lambda r:r.update(ended_at='2026-10-04T10:00:00Z'))
    elif defect=='naive_time':alter(scope,'runner',receipt=lambda r:r.update(ended_at='2026-10-04T09:05:00'))
    elif defect=='wrong_profile':alter(scope,'standard',body=lambda b:b.replace('999','888'))
    elif defect=='wrong_entry':alter(scope,'standard',body=lambda b:b.replace('data-runner-id="123"','data-runner-id="456"'))
    elif defect=='wrong_lazy':alter(scope,'expert',body=lambda b:b.replace('/runner/123/expert','/runner/456/expert'))
    elif defect=='detail_identity':alter(scope,'runner_expert',body=lambda b:b+'<b data-dog-id="888"></b>')
    elif defect=='retry':alter(scope,'runner',receipt=lambda r:r.update(retry_headers={'retry-after':'60'}))
    else:alter(scope,'expert',item=lambda item:item.update(url=URL.split('?')[0]+'/foreign/expert-form'))
    with pytest.raises(ValueError):audit_manifest(ref(scope))


def test_conflicting_duplicate_excludes_both_interpretations(scope):
    alter(scope,'runner',body=lambda _:history(duplicate=True,conflict=True))
    out=audit_manifest(ref(scope));assert out['normal']['unique_identity_rows']==4
    assert out['normal']['exclusion_categories']=={'DUPLICATE_EVENT_CONFLICT_OR_UNMARKED':1}
    assert out['shared_event_observations']==4


@pytest.mark.parametrize('kind,reason',[('future','EVENT_DATE_OR_URL_INVALID'),('unknown','EVENT_IDENTITY_INCOMPLETE')])
def test_future_or_unknown_event_never_qualifies(scope,kind,reason):
    alter(scope,'runner',body=lambda _:history(**{kind:True}))
    out=audit_manifest(ref(scope));assert out['normal']['unique_identity_rows']==4
    assert out['normal']['exclusion_categories']=={reason:1}


def test_retained_expert_local_failure_not_rewritten_as_success(scope):
    alter(scope,'expert',receipt=lambda r:r.update(status='FAILED_PRESERVED',error_type='TypeError'))
    assert audit_manifest(ref(scope))['preserved_expert_local_failure'] is True


def test_default_off_does_not_read_manifest(scope,monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',['audit','--manifest','/does/not/exist'])
    assert main()==0 and 'DEFAULT_OFF' in capsys.readouterr().out


def test_existing_output_is_not_overwritten(scope,tmp_path,monkeypatch,capsys):
    output=tmp_path/'out.json';output.write_text('original')
    monkeypatch.setattr('sys.argv',['audit','--execute','--manifest',str(scope),
        '--manifest-sha256',ref(scope)['sha256'],'--output',str(output)])
    assert main()==2 and output.read_text()=='original'
    assert '21.99' not in capsys.readouterr().out


def test_retained_structure_selects_history_not_second_best_time_table(scope):
    def retained_shape(body):
        body=body.replace('</tr></thead>', '<th></th><th></th></tr></thead>')
        body=body.replace('</td></tr>', '</td><td></td><td></td></tr>')
        body=body.replace('</tbody></table>', '<tr><td colspan="10">layout control</td></tr></tbody></table>')
        return body + '<table class="runner-form"><thead><tr><th>DATE</th><th>TIME</th></tr></thead><tbody><tr><td>summary</td><td>unused</td></tr></tbody></table>'
    alter(scope,'runner',body=retained_shape)
    out=audit_manifest(ref(scope))
    assert out['normal']['rendered_rows']==6 and out['normal']['unique_identity_rows']==5
    assert out['normal']['exclusion_categories']=={'NON_HISTORY_OR_INCOMPLETE_ROW':1}
