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

V3 = 'frozen_exact_source_title_venue_date_race_v3'
ANGLE_URL = 'https://www.thedogs.com.au/racing/angle-park/2026-07-20/4/test?trial=false'
ANGLE_RACE = 'Race 4 - AP_K - 2026-07-20'


def policy_fixture(tmp_path):
    forecasts = [{'race_id': ANGLE_RACE, 'prediction_id': 'p-angle'}]
    forecasts += [{'race_id': 'Race '+str(i)+' - SAN - 2026-07-20', 'prediction_id': 'p-'+str(i)} for i in range(1,210)]
    population = tmp_path/'population.json'
    population.write_text(json.dumps({'records':forecasts}))
    policy = {'schema':'frozen_offline_source_title_projection_policy_v3',
              'status':'FROZEN_BEFORE_ADDITIONAL_RESULT_DECODING',
              'population_races':210, 'native_venues':2, 'exact_display_entries':3,
              'field_policy_unchanged':True, 'source_hold_preserved':True,
              'new_runtime_or_provider_authority':False,
              'population_reference':{'path':str(population),'sha256':c.digest(population)},
              'mappings':[{'native_venue':'AP_K','official_url_slug':'angle-park',
                           'proposed_exact_title_venues':['Angle Park']},
                          {'native_venue':'SAN','official_url_slug':'sandown',
                           'proposed_exact_title_venues':['Sandown','Sandown Park']}]}
    path = tmp_path/'policy.json';path.write_text(json.dumps(policy))
    return path, forecasts, c.load_projection_policy(path,c.digest(path),forecasts)


def v3_declared(markup, policy, url=ANGLE_URL, race=ANGLE_RACE):
    from bs4 import BeautifulSoup
    return c.check_declared_identity(BeautifulSoup(markup,'html.parser'),url,V3,
                                    projection_policy=policy,race_id=race)


def test_frozen_v3_angle_park_exact_triple_without_generic_albion_alias(tmp_path):
    _,_,policy=policy_fixture(tmp_path)
    title='<title>thedogs - Angle Park 20 July 2026 Race 4</title>'
    assert v3_declared(title,policy)==V3
    for name in ('Albion Park (K)','AP_K','Angle-Park','angle park','Angle Park Extra'):
        with pytest.raises(ValueError):
            v3_declared(title.replace('Angle Park',name),policy)
    with pytest.raises(ValueError):
        v3_declared(title,policy,ANGLE_URL.replace('angle-park','albion-park'))
    with pytest.raises(ValueError):
        v3_declared(title,policy,race=ANGLE_RACE.replace('AP_K','SAN'))


@pytest.mark.parametrize('name',['Sandown','Sandown Park'])
def test_frozen_v3_sandown_exact_source_backed_variants(tmp_path,name):
    _,_,policy=policy_fixture(tmp_path)
    markup='<title>thedogs - '+name+' 20 July 2026 Race 3</title>'
    assert v3_declared(markup,policy,ANGLE_URL.replace('angle-park','sandown').replace('/4/','/3/'),'Race 3 - SAN - 2026-07-20')==V3


@pytest.mark.parametrize('change',[
    lambda text:text.replace('Race 4','Race 5'),
    lambda text:text.replace('20 July','21 July'),
    lambda text:text.replace('20 July','31 February'),
    lambda text:text.replace('July','Jul'),
    lambda text:text.replace('2026','2025'),
    lambda text:text+' extra',
])
def test_v3_title_anchoring_date_and_race_fail_closed(tmp_path,change):
    _,_,policy=policy_fixture(tmp_path)
    with pytest.raises(ValueError):
        v3_declared('<title>'+change('thedogs - Angle Park 20 July 2026 Race 4')+'</title>',policy)


@pytest.mark.parametrize('extra',[
    '<title>thedogs - Angle Park 20 July 2026 Race 4</title>',
    '<link rel="canonical">',
    '<meta property="og:url" content="">',
    '<link rel="canonical" href="'+ANGLE_URL.replace('/4/','/5/')+'">',
    '<meta property="og:url" content="'+ANGLE_URL.replace('angle-park','sandown')+'">',
    '<link rel="canonical" href="'+ANGLE_URL+'">',
])
def test_v3_not_general_bypass_for_ambiguous_or_present_declarations(tmp_path,extra):
    _,_,policy=policy_fixture(tmp_path)
    with pytest.raises(ValueError):
        v3_declared('<title>thedogs - Angle Park 20 July 2026 Race 4</title>'+extra,policy)


def test_v3_requires_external_policy_before_accepting_title(tmp_path):
    with pytest.raises(ValueError,match='policy'):
        v3_declared('<title>thedogs - Angle Park 20 July 2026 Race 4</title>',None)
    path,forecasts,_=policy_fixture(tmp_path)
    with pytest.raises(ValueError,match='hash'):
        c.load_projection_policy(path,'0'*64,forecasts)
    changed=list(forecasts);changed[0]={'race_id':ANGLE_RACE,'prediction_id':'reconstructed'}
    with pytest.raises(ValueError,match='population'):
        c.load_projection_policy(path,c.digest(path),changed)
    population=tmp_path/'population.json';population.write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        c.load_projection_policy(path,c.digest(path),forecasts)


def test_policy_changed_scope_or_duplicate_mapping_rejected(tmp_path):
    path,forecasts,_=policy_fixture(tmp_path)
    document=json.loads(path.read_text())
    document['mappings'].append(document['mappings'][0])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='mapping'):
        c.load_projection_policy(path,c.digest(path),forecasts)


def test_ready_v3_census_cannot_reclassify_held_completion():
    args=list(census());p=args[4][0]
    p['projection_policy_fallback']={'status':'QUALIFIED','original_parser_failure':'source_declared_race_identity_missing_or_mismatched'}
    args[1][0]['completion_observations']=[{'status':'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED','source_hold':False,'charged_attempts':1,'provider_requests':1}]
    assert c.check_census(*args)['strict_races']==1
    args[1][0]['completion_observations'][0]['source_hold']=True
    with pytest.raises(ValueError,match='V3.*READY'):
        c.check_census(*args)


def v3_source_fixture(tmp_path):
    row=fixture(tmp_path)
    _,_,policy=policy_fixture(tmp_path)
    body=tmp_path/'body'
    raw=body.read_bytes().replace(('<link rel="canonical" href="'+URL+'">').encode(),
                                 b'<title>thedogs - Angle Park 20 July 2026 Race 4</title>')
    body.write_bytes(raw)
    metadata=json.loads((tmp_path/'metadata.json').read_text())
    metadata.update(requested_url=ANGLE_URL,final_url=ANGLE_URL,response_sha256=c.digest(body))
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    row.update(race_id=ANGLE_RACE,prediction_id='p-angle')
    evidence=row['result_evidence'];reference=policy['reference']
    evidence['response_provenance'].update(response_sha256=c.digest(body),metadata_sha256=c.digest(tmp_path/'metadata.json'),projection_policy_reference=reference)
    evidence['projection_policy_fallback']={'status':'QUALIFIED','original_parser_failure':'source_declared_race_identity_missing_or_mismatched','projection_policy_reference':reference}
    evidence['races'][0]['source_identity']={'identity_semantics':V3,'source_title':'thedogs - Angle Park 20 July 2026 Race 4','source_title_identity':['angle-park','2026-07-20',4],'source_title_native_venue':'AP_K','source_declared_identities':[],'projection_policy_reference':reference}
    return row,policy


def test_complete_v3_source_check_preserves_existing_field_and_terminal_checks(tmp_path):
    row,policy=v3_source_fixture(tmp_path)
    assert c.check_source_identity(row,ANGLE_URL,policy)['identity_rule']==V3
    row['result_evidence']['runners'][1]['dog_name']='Invented'
    with pytest.raises(ValueError,match='identity or terminal'):
        c.check_source_identity(row,ANGLE_URL,policy)


@pytest.mark.parametrize('change',[
    lambda r:r.update(prediction_id='replay'),
    lambda r:r.update(race_id='Race 999 - AP_K - 2026-07-20'),
    lambda r:r['result_evidence']['projection_policy_fallback'].update(projection_policy_reference={'path':'elsewhere','sha256':'0'*64}),
    lambda r:r['result_evidence']['response_provenance'].update(projection_policy_reference={'path':'elsewhere','sha256':'0'*64}),
    lambda r:r['result_evidence'].update(original_transport_hold_preserved=True),
])
def test_v3_policy_scope_pin_and_hold_fail_before_body_decode(tmp_path,monkeypatch,change):
    row,policy=v3_source_fixture(tmp_path);change(row)
    def must_not_decode(_):
        pytest.fail('outcome body decoded before scope/policy rejection')
    monkeypatch.setattr(c,'source_rows',must_not_decode)
    with pytest.raises(ValueError):c.check_source_identity(row,ANGLE_URL,policy)


@pytest.mark.parametrize('change',[
    lambda r:r['result_evidence']['races'][0]['source_identity'].update(source_title_native_venue='SAN'),
    lambda r:r['result_evidence']['races'][0]['source_identity'].update(source_title_identity=['angle-park','2026-07-20',5]),
    lambda r:r['result_evidence']['races'][0]['source_identity'].update(source_title='fabricated'),
])
def test_v3_projection_claim_must_equal_independent_title(tmp_path,change):
    row,policy=v3_source_fixture(tmp_path);change(row)
    with pytest.raises(ValueError,match='independent title'):
        c.check_source_identity(row,ANGLE_URL,policy)


def test_policy_native_scope_uses_exact_native_labels_not_allocation_aliases(tmp_path):
    path,forecasts,_=policy_fixture(tmp_path)
    document=json.loads(path.read_text());document['mappings'][0]['native_venue']='AP'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='native mapping'):
        c.load_projection_policy(path,c.digest(path),forecasts)
