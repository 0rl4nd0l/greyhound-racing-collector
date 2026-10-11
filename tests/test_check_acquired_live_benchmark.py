"""Independent checks reject source/field/census drift without acquired outcomes."""
import copy
import hashlib
import json
import pytest
from scripts import check_acquired_live_benchmark as c

URL = 'https://www.thedogs.com.au/racing/healesville/2026-07-17/7/test?trial=false'


def fixture(tmp_path, rows=None):
    rows = rows or [(1, 'Alpha', '1st'), (2, 'Beta', 'DNF'), (9, 'Unused', 'SCR')]
    raw = ('<link rel="canonical" href="'+URL+'"><table class="race-runners--result">'+''.join(
        f'<tr class="race-runner"><td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td><td class="race-runners__name">{name}</td><td class="race-runners__finish-position">{place}</td></tr>'
        for box, name, place in rows)+'</table>').encode()
    body=tmp_path/'body'; body.write_bytes(raw)
    meta=tmp_path/'metadata.json'; meta.write_text(json.dumps({'requested_url':URL,'final_url':URL,'http_status':200,'response_sha256':hashlib.sha256(raw).hexdigest()}))
    _, native=c.source_rows(raw)
    runners=[{'box_number':r['box_number'],'original_source_box':r['box_number'],'dog_name':r['dog_name'],'finish_position':r['finish_position'],'terminal_status':r['status'] or 'FINISHED','is_winner':int(r['finish_position']==1)} for r in native if r['status']!='SCR']
    return {'race_id':'r','official_runner_identity_observed':True,'result_artifacts':[],
            'result_evidence':{'runner_name_origin':'SOURCE_NATIVE_OFFICIAL_HTML','response_provenance':{'response_path':str(body),'response_sha256':c.digest(body),'metadata_path':str(meta),'metadata_sha256':c.digest(meta)},'races':[{'native_official_rows':native}],'runners':runners}}


def test_source_identity_dnf_and_unused_reserve(tmp_path):
    assert c.check_source_identity(fixture(tmp_path), URL)['source_runners']==3


@pytest.mark.parametrize('change,reason', [
    (lambda r:r['result_evidence']['runners'][1].update(dog_name='Invented'),'identity or terminal'),
    (lambda r:r['result_evidence']['runners'][1].update(finish_position=2),'identity or terminal'),
    (lambda r:r['result_evidence']['runners'].pop(),'silently dropped'),
    (lambda r:r['result_evidence']['races'][0]['native_official_rows'][1].update(dog_name='Copied'),'independently parsed'),
    (lambda r:r.update(official_runner_identity_observed=False),'identity missing'),
])
def test_source_must_independently_confirm_every_original_runner(tmp_path,change,reason):
    row=fixture(tmp_path);change(row)
    with pytest.raises(ValueError,match=reason):c.check_source_identity(row,URL)


def test_raw_body_tampering_rejected(tmp_path):
    row=fixture(tmp_path);(tmp_path/'body').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash changed'):c.check_source_identity(row,URL)


def test_scope_route_and_unparsed_extra_source_row_rejected(tmp_path):
    row=fixture(tmp_path)
    with pytest.raises(ValueError,match='metadata mismatch'):c.check_source_identity(row,URL.replace('/7/','/8/'))
    raw=(tmp_path/'body').read_bytes().replace(b'</table>',b'<tr class="race-runner"><td>extra</td></tr></table>')
    with pytest.raises(ValueError,match='cell missing'):c.source_rows(raw)


def test_explicit_reserve_note_preserves_identity(tmp_path):
    row=fixture(tmp_path,[(1,'Alpha','1st'),(2,'Former','SCR'),(9,'Beta (from box 2)','2nd')])
    row['result_evidence']['runners'][1].update(box_number=2,dog_name='Beta')
    assert c.check_source_identity(row,URL)['forecast_runners']==2
    row['result_evidence']['runners'][1]['box_number']=3
    with pytest.raises(ValueError,match='continuity'):c.check_source_identity(row,URL)


def census():
    f=[{'race_id':'r','prediction_id':'p'}];projection={'race_id':'r'}
    rows=[{'race_id':'r','prediction_id':'p','model_version':'v:'+v,'official_runner_identity_observed':True,'result_evidence':projection} for v in ('full','half')]
    d=[{'race_id':'r','selected_projection':'ACQUIRED','disposition':'NEW_STRICT_RESULT'}]
    return f,d,rows,[],[projection]


def test_nonzero_strict_census_supported():
    assert c.check_census(*census())['strict_races']==1


@pytest.mark.parametrize('mutation,reason',[
    (lambda a:a[1].clear(),'census'),
    (lambda a:a[2].pop(),'variant'),
    (lambda a:a[3].extend(copy.deepcopy(a[2])),'classification'),
    (lambda a:a[2][0].update(prediction_id='alternate'),'primary'),
    (lambda a:a[4].clear(),'projection census'),
])
def test_census_drift_is_rejected(mutation,reason):
    args=list(census());mutation(args)
    with pytest.raises(ValueError,match=reason):c.check_census(*args)


def scored_record(day='2026-07-20'):
    return {'race_id':'race-'+day,'date':day,'prediction_id':'p-'+day,'model_version':'v:full',
            'allocation_status':'AUTHORISED_NONRESERVED','forecast_type':'ORIGINAL_SEALED_LIVE',
            'official_runner_identity_observed':True,'field_status':'EXACT_UNCHANGED','verified':True,
            'quote_at':day+'T00:00:00Z','cutoff_at':day+'T00:01:00Z','predicted_at':day+'T00:01:00Z',
            'sealed_at':day+'T00:02:00Z','jump_at':day+'T00:05:00Z','result_at':day+'T00:10:00Z',
            'runners':[{'runner_id':'a','box':1,'probability':.7,'decimal_odds':2.,'winner':1},
                       {'runner_id':'b','box':2,'probability':.3,'decimal_odds':3.,'winner':0}]}


def test_nonzero_metrics_and_calibration_verified_independently():
    # Production scorer creates only a synthetic claim; checker imports no scorer.
    from scripts.score_live_benchmark import benchmark
    rows=[scored_record(),scored_record('2026-07-21')]
    claim=benchmark(rows)
    check=c.independent.compare_scorecard(rows,claim)
    assert check['by_version']['v:full']['races']==2
    assert check['by_version']['v:full']['model']['top_accuracy']==1
    claim['by_version']['v:full']['model_calibration']['bins'][7]['observed_frequency']=0
    with pytest.raises(ValueError,match='metric mismatch'):
        c.independent.compare_scorecard(rows,claim)


def test_selected_and_market_control_missingness_preserved():
    from scripts.score_live_benchmark import benchmark
    row=scored_record();rid=row['race_id']
    frozen={'model_version':'v:full','selector':'model_confidence','target_coverage':.5,
            'selected_race_ids':[rid,'unresolved'],'market_confidence_race_ids':['unresolved',rid]}
    claimed={**frozen,'selected_outcomes':1,'selected_missing_outcomes':1,
             'market_selector_outcomes':1,'market_selector_missing_outcomes':1,
             'selected':benchmark([row]),'market_confidence':benchmark([row])}
    check=c.independent.check_primary_selector_closure([row],{'comparisons':[frozen]},{'comparisons':[claimed]})
    assert check['outcome_counts'][0]['selected_outcomes']==1
    claimed['selected_missing_outcomes']=0
    with pytest.raises(ValueError,match='outcome coverage'):
        c.independent.check_primary_selector_closure([row],{'comparisons':[frozen]},{'comparisons':[claimed]})
