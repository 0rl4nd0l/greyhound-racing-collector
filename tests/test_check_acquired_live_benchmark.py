"""Independent checks reject source/field/census drift without acquired outcomes."""
import copy
import hashlib
import json
import pytest
from scripts import check_acquired_live_benchmark as c

URL = 'https://www.thedogs.com.au/racing/healesville/2026-07-17/7/test?trial=false'
TITLE = 'thedogs - Healesville 17 July 2026 Race 7'


def declared(markup, url=URL, required_rule=None):
    from bs4 import BeautifulSoup
    return c.check_declared_identity(BeautifulSoup(markup, 'html.parser'), url, required_rule)


def test_existing_other_venue_canonical_path_preserved():
    other = URL.replace('healesville', 'bendigo')
    html = '<title>thedogs - Bendigo 17 July 2026 Race 7</title><link rel="canonical" href="'+other+'">'
    assert declared(html, other) == 'canonical_or_og_url'
    with pytest.raises(ValueError,match='unsupported'):
        declared('<title>thedogs - Bendigo 17 July 2026 Race 7</title>',other)


def test_title_adapter_cannot_bypass_title_with_matching_canonical():
    html = '<title>'+TITLE.replace('Race 7','Race 8')+'</title><link rel="canonical" href="'+URL+'">'
    with pytest.raises(ValueError,match='title identity mismatch'):
        declared(html,required_rule='exact_source_title_venue_date_race_v1')


def test_exact_source_title_can_independently_identify_missing_canonical():
    assert declared('<title>'+TITLE+'</title>') == 'exact_source_title_venue_date_race_v1'


@pytest.mark.parametrize('title', [TITLE.replace('Healesville','HEA'), TITLE.replace('17','18'),
    TITLE.replace('Race 7','Race 8'), TITLE.replace('July','February').replace('17','31'),
    TITLE.replace('2026','2025'), TITLE.replace('July','Jul'), TITLE+' trial',
    TITLE.replace('Healesville','healesville')])
def test_source_title_wrong_or_unsupported_identity_rejected(title):
    with pytest.raises(ValueError): declared('<title>'+title+'</title>')


@pytest.mark.parametrize('markup', ['', '<title>'+TITLE+'</title><title>'+TITLE+'</title>',
    '<title>'+TITLE+'</title><link rel="canonical">',
    '<title>'+TITLE+'</title><meta property="og:url" content="">',
    '<title>'+TITLE+'</title><link rel="canonical" href="'+URL.replace('/7/','/8/')+'">'])
def test_missing_ambiguous_or_contradictory_identity_rejected(markup):
    with pytest.raises(ValueError): declared(markup)


def test_title_does_not_bypass_trial_or_domain_boundary():
    for url in (URL.replace('trial=false','trial=true'), URL.replace('www.thedogs.com.au','example.com')):
        with pytest.raises(ValueError): declared('<title>'+TITLE+'</title>',url)


def test_multiple_native_result_tables_rejected():
    with pytest.raises(ValueError,match='table missing or ambiguous'):
        c.source_rows('<table class="race-runners--result"></table>'*2)


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


def test_native_race_identity_rule_requires_title_even_with_canonical(tmp_path):
    row=fixture(tmp_path)
    row['result_evidence']['races'][0]['source_identity']={'identity_semantics':'exact_source_title_venue_date_race_v1'}
    raw=('<title>'+TITLE.replace('Race 7','Race 8')+'</title>').encode()+(tmp_path/'body').read_bytes()
    (tmp_path/'body').write_bytes(raw)
    metadata=json.loads((tmp_path/'metadata.json').read_text())
    metadata['response_sha256']=c.digest(tmp_path/'body')
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    row['result_evidence']['response_provenance'].update(response_sha256=c.digest(tmp_path/'body'),metadata_sha256=c.digest(tmp_path/'metadata.json'))
    with pytest.raises(ValueError,match='title identity mismatch'):
        c.check_source_identity(row,URL)


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


def test_reviewed_strict_preserves_hold_and_consumption():
    args=list(census())
    args[1][0].update(disposition='REVIEWED_OFFLINE_STRICT_RESULT',completion_observations=[
        {'source_hold':True,'charged_attempts':1,'provider_requests':1,'status':'TERMINAL_SOURCE_HOLD_NO_RETRY'}])
    args[4][0].update(original_transport_hold_preserved=True,reviewed_reclassification={'race_id':'r'})
    assert c.check_census(*args)['strict_races']==1
    args[1][0]['completion_observations'][0]['source_hold']=False
    with pytest.raises(ValueError,match='terminal accounting'): c.check_census(*args)


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
