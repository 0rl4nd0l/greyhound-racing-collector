"""Independent checks reject source/field/census drift without acquired outcomes."""
import copy
import hashlib
import json
import pytest
from scripts import check_acquired_live_benchmark as c

URL = 'https://www.thedogs.com.au/racing/healesville/2026-07-17/7/test?trial=false'
TITLE = 'thedogs - Healesville 17 July 2026 Race 7'


def prereceipt(tmp_path):
    sidecar = {'prejump_shadow_metadata': {'status': 'PASS', 'fail_reasons': [],
        'metadata_is_leakage_safe': True, 'metadata_captured_at': '2026-07-17T01:00:00Z',
        'runner_box_name_list': [{'box_number': 1, 'dog_name': 'Alpha'}, {'box_number': 2, 'dog_name': 'Beta'}]},
        'canonical_runner_alignment': {'status': 'aligned', 'canonical_runner_set_status': 'available',
            'duplicate_source_runner_names': [], 'missing_canonical_participants': [],
            'dropped_participants': [{'box_number': 3, 'dog_name': 'Unused'}], 'remapped_participants': []}}
    path = tmp_path/'pre.json'; path.write_text(json.dumps(sidecar))
    ref = {'path': str(path), 'sha256': c.digest(path)}
    original = {'original': {'race_id': 'r', 'record_key': 'p', 'score_timestamp': '2026-07-17T01:02:00Z',
        'jump_timestamp': '2026-07-17T01:10:00Z', 'predictions': sidecar['prejump_shadow_metadata']['runner_box_name_list']},
        'append_errors': [], 'append_proof': {'sealed_at': '2026-07-17T01:02:01Z', 'artifacts': {'sidecar': ref},
            'native_prediction': {'input_hashes': {'sidecar_sha256': ref['sha256']}}},
        'win_field': {'status': 'EXACT_COMPLETE_WIN_FIELD', 'capture_at': '2026-07-17T01:00:30Z',
            'available_at': '2026-07-17T01:01:00Z'}}
    return original, sidecar


def test_v6_independent_pre_race_authentication(tmp_path):
    original, sidecar = prereceipt(tmp_path)
    assert c.check_pre_race_field_proof(original)['pre_race_active_field'] == sidecar['prejump_shadow_metadata']['runner_box_name_list']


@pytest.mark.parametrize('change', [
    lambda o: o['append_proof']['native_prediction']['input_hashes'].update(sidecar_sha256='0'*64),
    lambda o: o['original'].update(score_timestamp='2026-07-17T00:59:00Z'),
    lambda o: o['original'].update(jump_timestamp='2026-07-17T01:02:00Z'),
    lambda o: o['win_field'].update(status='PARTIAL'),
    lambda o: o['win_field'].update(available_at='2026-07-17T01:03:00Z'),
    lambda o: o['original']['predictions'].pop(),
    lambda o: o.update(append_errors=['unsealed']),
])
def test_v6_unsealed_late_partial_or_changed_pre_race_proof_rejected(tmp_path, change):
    original, _ = prereceipt(tmp_path); change(original)
    with pytest.raises(ValueError): c.check_pre_race_field_proof(original)


@pytest.mark.parametrize('change', [
    lambda s: s['prejump_shadow_metadata'].update(status='FAIL'),
    lambda s: s['prejump_shadow_metadata'].update(metadata_is_leakage_safe=False),
    lambda s: s['prejump_shadow_metadata'].update(fail_reasons=['bad']),
    lambda s: s['canonical_runner_alignment'].update(missing_canonical_participants=['Alpha']),
    lambda s: s['canonical_runner_alignment'].update(duplicate_source_runner_names=['Alpha']),
])
def test_v6_invalid_pre_race_sidecar_rejected_even_if_hash_matches(tmp_path, change):
    original, sidecar = prereceipt(tmp_path); change(sidecar)
    path=tmp_path/'pre.json'; path.write_text(json.dumps(sidecar)); sha=c.digest(path)
    original['append_proof']['artifacts']['sidecar']['sha256']=sha
    original['append_proof']['native_prediction']['input_hashes']['sidecar_sha256']=sha
    with pytest.raises(ValueError): c.check_pre_race_field_proof(original)


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


def name_cell(name='Alpha', dog_id='123', badge='NBT', note=''):
    return (f'<td class="race-runners__name"><blackbook-dog data-dog-id="{dog_id}"></blackbook-dog>'
            f'<div class="race-runners__name__dog"><a href="/dogs/{dog_id}/alpha">{name}</a>'
            f'<span class="race-runners__name__time">{badge}</span>'
            f'<span class="race-runners__name__box">{note}</span></div>'
            '<div>T: Trainer R/T:</div></td>')


def parse_name(markup):
    from bs4 import BeautifulSoup
    return c.structured_source_name(BeautifulSoup(markup,'html.parser').find('td'),1)


def test_v4_name_is_exact_anchor_not_time_badge_or_trainer():
    name,witness=parse_name(name_cell())
    assert name=='Alpha'
    assert witness['source_dog_id']=='123'
    assert witness['source_blackbook_ids']==['123']
    assert witness['separate_time_annotations']==['NBT']
    assert witness['original_parser_name']=='Alpha NBT'
    assert parse_name(name_cell(name='Alpha NBT'))[0]=='Alpha NBT'
    assert parse_name(name_cell(note='(from box 2)'))[0]=='Alpha (from box 2)'


@pytest.mark.parametrize('change',[
    lambda x:x.replace('/dogs/123/alpha','https://evil.example/dogs/123/alpha'),
    lambda x:x.replace('/dogs/123/alpha','//www.thedogs.com.au/dogs/123/alpha'),
    lambda x:x.replace('/dogs/123/alpha','/dogs/0/alpha'),
    lambda x:x.replace('/dogs/123/alpha','/dogs/123/alpha?other=1'),
    lambda x:x.replace('/dogs/123/alpha','/dogs/123/alpha#part'),
    lambda x:x.replace('data-dog-id="123"','data-dog-id="456"'),
    lambda x:x.replace('</a>','</a><a href="/dogs/123/alpha">Alpha</a>'),
    lambda x:x.replace('>Alpha</a>','></a>'),
    lambda x:x.replace('</blackbook-dog>','</blackbook-dog><blackbook-dog data-dog-id="123"></blackbook-dog>'),
    lambda x:x.replace('<span class="race-runners__name__time">','<span class="race-runners__name__time">1</span><span class="race-runners__name__time">'),
    lambda x:x.replace('class="race-runners__name__box">','class="race-runners__name__box">from box 2'),
])
def test_v4_ambiguous_foreign_or_unbound_dog_identity_rejected(change):
    with pytest.raises(ValueError):parse_name(change(name_cell()))


def name_policy_fixture(tmp_path):
    title_path,forecasts,title_policy=policy_fixture(tmp_path)
    review=tmp_path/'review.json';review.write_text('{}')
    parser=tmp_path/'pinned-parser.py';parser.write_text('# frozen parser\n')
    doc={'schema':'frozen_offline_structured_dog_name_policy_v4',
         'status':'FROZEN_BEFORE_CORRECTED_METRICS','field_and_scoring_policy_unchanged':True,
         'newly_corrected_metrics_inspected':False,'no_new_provider_or_runtime_authority':True,
         'population_races':210,'population_reference':title_policy['document']['population_reference'],
         'independent_review_reference':{'path':str(review),'sha256':c.digest(review)},
         'source_title_policy_reference':title_policy['reference'],
         'prior_parser_references':[{'path':str(parser),'sha256':c.digest(parser)}],
         'terminal_policy':{'accepted_vocabulary_unchanged':True,'no_runner_removal_or_renormalization':True,
                            **{code:'UNMAPPED; preserved' for code in ('F','T','-')}},
         'version_and_scope':{k:True for k in ('base_v2_v3_projector_bytes_preserved','external_policy_hash_required',
          'no_model_fit_calibration_selector_threshold_change','no_performance_based_rule',
          'no_provider_or_runtime_authority','separate_v4_adapter_required')}}
    path=tmp_path/'name-policy.json';path.write_text(json.dumps(doc))
    return path,forecasts,title_policy,c.load_runner_name_policy(path,c.digest(path),forecasts,title_policy)


def test_v4_policy_pin_scope_and_original_parser_bytes_verified(tmp_path):
    path,forecasts,title_policy,policy=name_policy_fixture(tmp_path)
    with pytest.raises(ValueError,match='hash'):
        c.load_runner_name_policy(path,'0'*64,forecasts,title_policy)
    changed=copy.deepcopy(forecasts);changed[0]['prediction_id']='replacement'
    with pytest.raises(ValueError,match='population'):
        c.load_runner_name_policy(path,c.digest(path),changed,title_policy)
    (tmp_path/'pinned-parser.py').write_text('# changed')
    with pytest.raises(ValueError,match='linked hash'):
        c.load_runner_name_policy(path,c.digest(path),forecasts,title_policy)


@pytest.mark.parametrize('code',['F','T','-'])
def test_v4_policy_does_not_authorize_new_terminal_meaning(tmp_path,code):
    path,forecasts,title_policy,_=name_policy_fixture(tmp_path)
    doc=json.loads(path.read_text());doc['terminal_policy'][code]='DNF'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='terminal'):
        c.load_runner_name_policy(path,c.digest(path),forecasts,title_policy)
    markup='<table class="race-runners--result"><tr class="race-runner"><td class="race-runners__box"><i name="rug_1"></i></td>'+name_cell()+'<td class="race-runners__finish-position">'+code+'</td></tr></table>'
    with pytest.raises(ValueError,match='terminal'):
        c.source_rows(markup,structured_names=True)


def v4_source_fixture(tmp_path, title=False):
    from bs4 import BeautifulSoup
    row=fixture(tmp_path,[(1,'Alpha','1st'),(2,'Beta','DNF'),(9,'Unused','SCR')])
    _,_,title_policy,policy=name_policy_fixture(tmp_path)
    body=tmp_path/'body';soup=BeautifulSoup(body.read_bytes(),'html.parser')
    for node in soup.select('tr.race-runner'):
        old=node.select_one('td.race-runners__name');name=old.get_text();box=node.select_one('[name]')['name'][4:]
        old.replace_with(BeautifulSoup(name_cell(name=name,dog_id='10'+box),'html.parser').find('td'))
    evidence=row['result_evidence'];reference=policy['reference']
    row.update(race_id=ANGLE_RACE,prediction_id='p-angle')
    soup.find('link')['href']=ANGLE_URL
    identity={'identity_semantics':'exact_official_venue_slug_date_race_number'}
    if title:
        soup.find('link').decompose();title_node=soup.new_tag('title');title_node.string='thedogs - Angle Park 20 July 2026 Race 4';soup.insert(0,title_node)
        identity={'identity_semantics':V3,'source_title':title_node.string,'source_title_identity':['angle-park','2026-07-20',4],
          'source_title_native_venue':'AP_K','source_declared_identities':[],'projection_policy_reference':title_policy['reference']}
        evidence['projection_policy_fallback']={'status':'REJECTED','original_parser_failure':'source_declared_race_identity_missing_or_mismatched',
          'fallback_failure':'official_runner_name_changed_at_box:1','projection_policy_reference':title_policy['reference']}
    body.write_bytes(str(soup).encode())
    meta=tmp_path/'metadata.json';metadata=json.loads(meta.read_text());metadata.update(requested_url=ANGLE_URL,final_url=ANGLE_URL,response_sha256=c.digest(body));meta.write_text(json.dumps(metadata))
    _,rows=c.source_rows(body.read_bytes(),structured_names=True)
    evidence['races'][0].update(native_official_rows=rows,source_identity=identity,
      structured_runner_name_witnesses=[c.structured_source_name(n.select_one('td.race-runners__name'),r['box_number'])[1] for n,r in zip(soup.select('tr.race-runner'),rows)])
    evidence['runner_name_policy_fallback']={'status':'QUALIFIED','original_parser_failure':'official_runner_name_changed_at_box:1','runner_name_policy_reference':reference}
    evidence['response_provenance'].update(response_sha256=c.digest(body),metadata_sha256=c.digest(meta),runner_name_policy_reference=reference,projection_policy_reference=title_policy['reference'])
    return row,title_policy,policy


@pytest.mark.parametrize('title',[False,True])
def test_v4_preserves_canonical_without_title_and_v3_identity_with_original_rejection(tmp_path,title):
    row,title_policy,policy=v4_source_fixture(tmp_path,title)
    check=c.check_source_identity(row,ANGLE_URL,title_policy,policy)
    assert check['forecast_runners']==2
    assert check['identity_rule']==(V3 if title else 'canonical_or_og_url')


@pytest.mark.parametrize('change',[
    lambda r:r.update(prediction_id='newprediction'),
    lambda r:r['result_evidence']['runner_name_policy_fallback'].update(runner_name_policy_reference={'path':'elsewhere','sha256':'0'*64}),
    lambda r:r['result_evidence']['response_provenance'].update(runner_name_policy_reference={'path':'elsewhere','sha256':'0'*64}),
    lambda r:r['result_evidence'].update(original_transport_hold_preserved=True),
])
def test_v4_scope_policy_hold_rejected_before_body(tmp_path,monkeypatch,change):
    row,title_policy,policy=v4_source_fixture(tmp_path);change(row)
    def no_decode(*a,**kw):pytest.fail('body decoded before invalid policy or scope rejected')
    monkeypatch.setattr(c,'source_rows',no_decode)
    with pytest.raises(ValueError):c.check_source_identity(row,ANGLE_URL,title_policy,policy)


def test_v4_projected_name_witness_and_true_changed_names_rejected(tmp_path):
    row,title_policy,policy=v4_source_fixture(tmp_path)
    bad=copy.deepcopy(row);bad['result_evidence']['races'][0]['structured_runner_name_witnesses'][0]['source_dog_id']='999'
    with pytest.raises(ValueError,match='witnesses'):
        c.check_source_identity(bad,ANGLE_URL,title_policy,policy)
    row['result_evidence']['runners'][0]['dog_name']='Different'
    with pytest.raises(ValueError,match='identity or terminal'):
        c.check_source_identity(row,ANGLE_URL,title_policy,policy)


def test_v4_cannot_reclassify_held_completion():
    args=list(census());args[4][0]['runner_name_policy_fallback']={'status':'QUALIFIED'}
    args[1][0]['completion_observations']=[{'status':'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED','source_hold':False,'charged_attempts':1,'provider_requests':1}]
    assert c.check_census(*args)['strict_races']==1
    args[1][0]['completion_observations'][0]['source_hold']=True
    with pytest.raises(ValueError,match='V4.*READY'):c.check_census(*args)


def active_route_markup():
    route='/racing/angle-park/2026-07-20/4/test?trial=false'
    return ('<title>thedogs - Unmapped Sponsored Venue 20 July 2026 Race 4</title>'
      '<nav><a class="tabs__item--active" href="/racing">Racing</a></nav>'
      f'<a class="race-box--active" href="{route}"></a>'
      '<a class="dropdown__menu__meeting--active" href="/racing/angle-park/2026-07-20?trial=false"></a>'
      f'<div class="page__layout"><div class="tabs-wrapper"><a class="tabs__item--active" href="{route}">RESULTS</a></div></div>')


def active_identity(markup,policy):
    from bs4 import BeautifulSoup
    return c.check_active_route_identity(BeautifulSoup(markup,'html.parser'),ANGLE_URL,ANGLE_RACE,policy)


def test_v5_independent_exact_active_routes_do_not_infer_display_alias(tmp_path):
    _,_,policy=policy_fixture(tmp_path)
    proof=active_identity(active_route_markup(),policy)
    assert proof['active_race_number']==4
    assert proof['source_title'].startswith('thedogs - Unmapped Sponsored Venue')
    assert active_identity(active_route_markup().replace('href="/racing/','href="https://www.thedogs.com.au/racing/'),policy)


@pytest.mark.parametrize('change',[
    lambda x:x.replace('race-box--active','race-box'),
    lambda x:x+'<a class="race-box--active" href="/racing/wrong"></a>',
    lambda x:x.replace('class="page__layout"','class="nav"'),
    lambda x:x+'<div class="page__layout"><div class="tabs-wrapper"><a class="tabs__item--active" href="/racing/wrong">RESULTS</a></div></div>',
    lambda x:x.replace('dropdown__menu__meeting--active','meeting'),
    lambda x:x+'<a class="dropdown__menu__meeting--active" href="/racing/wrong"></a>',
    lambda x:x.replace('/4/test','/5/test'),
    lambda x:x.replace('angle-park','sandown'),
    lambda x:x.replace('trial=false','trial=true'),
    lambda x:x.replace('trial=false','trial=false&unexpected=1'),
    lambda x:x.replace('href="/racing/','href="//www.thedogs.com.au/racing/'),
    lambda x:x.replace('href="/racing/','href="https:/racing/'),
    lambda x:x.replace('href="/racing/','href="https://evil.example/racing/'),
    lambda x:x.replace('>RESULTS<','>PREVIEW<'),
    lambda x:x.replace('20 July','21 July'),
    lambda x:x.replace('20 July','31 February'),
    lambda x:x.replace('Race 4','Race 5'),
    lambda x:x.replace('Unmapped Sponsored Venue','Sandown'),
    lambda x:x+'<link rel="canonical">',
    lambda x:x+'<meta property="og:url" content="">',
    lambda x:x+'<link rel="canonical" href="'+ANGLE_URL+'">',
])
def test_v5_missing_ambiguous_or_contradictory_active_routes_reject(tmp_path,change):
    _,_,policy=policy_fixture(tmp_path)
    with pytest.raises(ValueError):active_identity(change(active_route_markup()),policy)


def structural_policy_fixture(tmp_path):
    _,forecasts,title_policy,name_policy=name_policy_fixture(tmp_path)
    doc={'schema':'frozen_offline_active_route_identity_policy_v5','status':'FROZEN_BEFORE_CORRECTED_METRICS',
         'field_and_scoring_policy_unchanged':True,'newly_corrected_metrics_inspected':False,
         'no_model_fit_calibration_selector_threshold_change':True,'no_new_provider_or_runtime_authority':True,
         'population_races':210,'population_reference':name_policy['document']['population_reference'],
         'independent_review_reference':name_policy['document']['independent_review_reference'],
         'prior_parser_references':name_policy['document']['prior_parser_references'],
         'source_title_policy_reference':title_policy['reference'],'runner_name_policy_reference':name_policy['reference'],
         'selectors':{'active_meeting':'a.dropdown__menu__meeting--active','active_race':'a.race-box--active',
                      'active_result_tab':'.page__layout .tabs-wrapper a.tabs__item--active'}}
    path=tmp_path/'structural-policy.json';path.write_text(json.dumps(doc))
    return path,forecasts,title_policy,name_policy,c.load_structural_identity_policy(path,c.digest(path),forecasts,title_policy,name_policy)


def test_v5_policy_authenticates_external_prior_policies_scope_and_hashes(tmp_path):
    path,forecasts,title_policy,name_policy,policy=structural_policy_fixture(tmp_path)
    with pytest.raises(ValueError,match='hash'):
        c.load_structural_identity_policy(path,'0'*64,forecasts,title_policy,name_policy)
    with pytest.raises(ValueError,match='external policy'):
        c.load_structural_identity_policy(path,c.digest(path),forecasts,None,name_policy)
    changed=copy.deepcopy(forecasts);changed[0]['prediction_id']='new'
    with pytest.raises(ValueError,match='population'):
        c.load_structural_identity_policy(path,c.digest(path),changed,title_policy,name_policy)
    doc=json.loads(path.read_text());doc['selectors']['active_result_tab']='a.tabs__item--active';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='selector scope'):
        c.load_structural_identity_policy(path,c.digest(path),forecasts,title_policy,name_policy)


def v5_source_fixture(tmp_path, names=False):
    from bs4 import BeautifulSoup
    row,title_policy,name_policy=v4_source_fixture(tmp_path)
    _,_,title_policy,name_policy,policy=structural_policy_fixture(tmp_path)
    body=tmp_path/'body';soup=BeautifulSoup(body.read_bytes(),'html.parser');soup.find('link').decompose()
    evidence=row['result_evidence']
    if not names:
        for n in soup.select('.race-runners__name__time'):n.string=''
        evidence.pop('runner_name_policy_fallback')
        evidence['races'][0].pop('structured_runner_name_witnesses')
        evidence['response_provenance'].pop('runner_name_policy_reference')
    body.write_bytes((active_route_markup()+str(soup)).encode())
    soup=BeautifulSoup(body.read_bytes(),'html.parser')
    meta=tmp_path/'metadata.json';metadata=json.loads(meta.read_text());metadata['response_sha256']=c.digest(body);meta.write_text(json.dumps(metadata))
    evidence['response_provenance'].update(response_sha256=c.digest(body),metadata_sha256=c.digest(meta),
                                          structural_identity_policy_reference=policy['reference'])
    _,rows=c.source_rows(body.read_bytes(),structured_names=names)
    evidence['races'][0]['native_official_rows']=rows
    evidence['races'][0]['source_identity']={**c.check_active_route_identity(soup,ANGLE_URL,ANGLE_RACE,title_policy),
      'identity_semantics':c.V5_IDENTITY,'projection_policy_reference':title_policy['reference'],
      'structural_identity_policy_reference':policy['reference']}
    if names:
        evidence['races'][0]['structured_runner_name_witnesses']=[c.structured_source_name(n.select_one('td.race-runners__name'),r['box_number'])[1] for n,r in zip(soup.select('tr.race-runner'),rows)]
    evidence['structural_identity_policy_fallback']={'status':'QUALIFIED','original_parser_failure':'source_title_display_slug_native_not_exact_policy_member','structural_identity_policy_reference':policy['reference']}
    evidence['projection_policy_fallback']={'status':'REJECTED','original_parser_failure':'source_declared_race_identity_missing_or_mismatched','fallback_failure':'source_title_display_slug_native_not_exact_policy_member','projection_policy_reference':title_policy['reference']}
    return row,title_policy,name_policy,policy


@pytest.mark.parametrize('names',[False,True])
def test_v5_full_source_identity_with_optional_frozen_name_repair(tmp_path,names):
    row,title_policy,name_policy,policy=v5_source_fixture(tmp_path,names)
    assert c.check_source_identity(row,ANGLE_URL,title_policy,name_policy,policy)['identity_rule']==c.V5_IDENTITY


@pytest.mark.parametrize('change',[
    lambda r:r.update(prediction_id='replacement'),
    lambda r:r['result_evidence']['structural_identity_policy_fallback'].update(structural_identity_policy_reference={'path':'elsewhere','sha256':'0'*64}),
    lambda r:r['result_evidence']['structural_identity_policy_fallback'].update(original_parser_failure='some_other_error'),
    lambda r:r['result_evidence']['projection_policy_fallback'].update(fallback_failure='some_other_error'),
    lambda r:r['result_evidence']['response_provenance'].update(structural_identity_policy_reference={'path':'elsewhere','sha256':'0'*64}),
    lambda r:r['result_evidence'].update(original_transport_hold_preserved=True),
])
def test_v5_scope_policy_and_failed_identity_chain_reject_before_body(tmp_path,monkeypatch,change):
    row,title_policy,name_policy,policy=v5_source_fixture(tmp_path);change(row)
    def no_decode(*a,**kw):pytest.fail('decoded before V5 policy/scope check')
    monkeypatch.setattr(c,'source_rows',no_decode)
    with pytest.raises(ValueError):c.check_source_identity(row,ANGLE_URL,title_policy,name_policy,policy)


def test_v5_projected_route_witness_must_match_independent_source(tmp_path):
    row,title_policy,name_policy,policy=v5_source_fixture(tmp_path)
    row['result_evidence']['races'][0]['source_identity']['active_race_href']='/racing/invented'
    with pytest.raises(ValueError,match='independent active source'):
        c.check_source_identity(row,ANGLE_URL,title_policy,name_policy,policy)


def test_v5_cannot_reopen_original_held_request():
    args=list(census());args[4][0]['structural_identity_policy_fallback']={'status':'QUALIFIED'}
    args[1][0]['completion_observations']=[{'status':'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED','source_hold':False,'charged_attempts':1,'provider_requests':1}]
    assert c.check_census(*args)['strict_races']==1
    args[1][0]['completion_observations'][0]['source_hold']=True
    with pytest.raises(ValueError,match='V5.*READY'):c.check_census(*args)


def v6_source_fixture(tmp_path, title=False):
    from bs4 import BeautifulSoup
    row,title_policy,name_policy=v4_source_fixture(tmp_path,title)
    original,sidecar=prereceipt(tmp_path)
    sidecar['prejump_shadow_metadata']['source_url']=ANGLE_URL
    sidecar['canonical_runner_alignment'].update(canonical_source_url=ANGLE_URL,canonical_runner_count=2,prediction_runner_count=2)
    (tmp_path/'pre.json').write_text(json.dumps(sidecar));sha=c.digest(tmp_path/'pre.json')
    original['append_proof']['artifacts']['sidecar']['sha256']=sha
    original['append_proof']['native_prediction']['input_hashes']['sidecar_sha256']=sha
    original['original'].update(race_id=row['race_id'],record_key=row['prediction_id'])
    reference={'path':'frozen-v6','sha256':'6'*64}
    policy={'reference':reference,'population':{row['race_id']:row['prediction_id']},
            'originals':{row['prediction_id']:original}}
    evidence=row['result_evidence'];evidence.pop('runner_name_policy_fallback')
    evidence['pre_race_field_policy_fallback']={'status':'QUALIFIED','original_parser_failure':'official_runner_outside_forecast_field',
                                              'pre_race_field_policy_reference':reference}
    if title: evidence['projection_policy_fallback']['fallback_failure']='official_runner_outside_forecast_field'
    body=tmp_path/'body';soup=BeautifulSoup(body.read_bytes(),'html.parser');soup.select_one('[name="rug_9"]')['name']='rug_3'
    body.write_bytes(str(soup).encode());meta=tmp_path/'metadata.json';md=json.loads(meta.read_text());md['response_sha256']=c.digest(body);meta.write_text(json.dumps(md))
    _,rows=c.source_rows(body.read_bytes(),structured_names=True)
    proof=c.check_pre_race_field_proof(original)
    evidence['races'][0].update(native_official_rows=rows,raw_native_official_rows=c.source_rows(body.read_bytes())[1],
        structured_runner_name_witnesses=[c.structured_source_name(n.select_one('td.race-runners__name'),r['box_number'])[1] for n,r in zip(soup.select('tr.race-runner'),rows)],
        pre_race_field_projection=c.check_pre_race_field_derivation(rows,proof))
    evidence['response_provenance'].update(response_sha256=c.digest(body),metadata_sha256=c.digest(meta),
        pre_race_field_policy_reference=reference,source_name_extraction_policy_reference=name_policy['reference'],
        sidecar_reference=proof['sidecar_reference'],runner_name_policy_reference=None)
    return row,title_policy,name_policy,policy


@pytest.mark.parametrize('title',[False,True])
def test_v6_independent_source_excludes_only_sealed_inactive_identity(tmp_path,title):
    row,title_policy,name_policy,policy=v6_source_fixture(tmp_path,title)
    assert c.check_source_identity(row,ANGLE_URL,title_policy,name_policy,None,policy)['source_runners']==3


@pytest.mark.parametrize('change',[
    lambda r:r['result_evidence']['races'][0]['pre_race_field_projection']['excluded_source_rows'].clear(),
    lambda r:r['result_evidence']['races'][0]['raw_native_official_rows'].pop(),
    lambda r:r['result_evidence']['pre_race_field_policy_fallback'].update(pre_race_field_policy_reference=None),
    lambda r:r['result_evidence']['response_provenance'].update(source_name_extraction_policy_reference=None),
    lambda r:r['result_evidence'].update(original_transport_hold_preserved=True),
])
def test_v6_rejects_tampered_witnesses_dependencies_and_hold(tmp_path,change):
    row,title_policy,name_policy,policy=v6_source_fixture(tmp_path);change(row)
    with pytest.raises(ValueError): c.check_source_identity(row,ANGLE_URL,title_policy,name_policy,None,policy)


def test_v6_original_active_scratch_cannot_be_excluded(tmp_path):
    original,_=prereceipt(tmp_path);proof=c.check_pre_race_field_proof(original)
    proof['dropped_participants'].append({'box_number':1,'dog_name':'Alpha'})
    with pytest.raises(ValueError,match='original scratched'):
        c.check_pre_race_field_derivation([{'box_number':1,'dog_name':'Alpha','status':'L/SCR','finish_position':None}],proof)


def test_v6_promoted_identity_requires_pre_race_and_source_note(tmp_path):
    original,_=prereceipt(tmp_path);proof=c.check_pre_race_field_proof(original)
    source=[{'box_number':10,'dog_name':'Beta (from box 2)','status':'DNF','finish_position':None}]
    with pytest.raises(ValueError,match='continuity'):
        c.check_pre_race_field_derivation(source,proof)
    proof['remapped_participants']=[{'source_box_number':10,'original_box_number':10,'final_box_number':2,'dog_name':'Beta'}]
    derived=c.check_pre_race_field_derivation(source,proof)
    assert derived['derived_original_field_rows'][0]['box_number']==2
    assert derived['derived_original_field_rows'][0]['status']=='DNF'
    proof['remapped_participants']*=2
    with pytest.raises(ValueError,match='continuity'):
        c.check_pre_race_field_derivation(source,proof)


def field_policy_fixture(tmp_path):
    _,forecasts,title,name,structural=structural_policy_fixture(tmp_path)
    original_rows=[{'original':{'race_id':r['race_id'],'record_key':r['prediction_id']}} for r in forecasts]
    original_rows.append({'original':{'race_id':forecasts[0]['race_id'],'record_key':'retained-nonprimary'}})
    originals=tmp_path/'originals.json';originals.write_text(json.dumps(original_rows))
    review=tmp_path/'field-review.json';review.write_text(json.dumps({'status':'PASS_CRITERIA_BEFORE_CORRECTED_METRICS','criteria':{'test':'frozen'}}))
    document={'schema':'frozen_offline_pre_race_field_policy_v6','status':'FROZEN_BEFORE_CORRECTED_METRICS',
        'field_and_scoring_policy_unchanged':True,'newly_corrected_metrics_inspected':False,
        'no_model_fit_calibration_selector_threshold_change':True,'no_new_provider_or_runtime_authority':True,
        'population_races':210,'population_reference':structural['document']['population_reference'],
        'authenticated_originals_reference':{'path':str(originals),'sha256':c.digest(originals)},
        'independent_review_reference':{'path':str(review),'sha256':c.digest(review)},
        'prior_parser_references':structural['document']['prior_parser_references'],
        'source_title_policy_reference':title['reference'],'runner_name_policy_reference':name['reference'],
        'structural_identity_policy_reference':structural['reference'],'criteria':{'test':'frozen'}}
    path=tmp_path/'field-policy.json';path.write_text(json.dumps(document))
    return path,forecasts,title,name,structural


def test_v6_policy_pin_scope_and_exact_primary_selection(tmp_path):
    path,forecasts,title,name,structural=field_policy_fixture(tmp_path)
    policy=c.load_pre_race_field_policy(path,c.digest(path),forecasts,title,name,structural)
    assert len(policy['originals'])==210 and 'retained-nonprimary' not in policy['originals']
    with pytest.raises(ValueError,match='hash'):
        c.load_pre_race_field_policy(path,'0'*64,forecasts,title,name,structural)
    with pytest.raises(ValueError,match='dependency'):
        c.load_pre_race_field_policy(path,c.digest(path),forecasts,title,None,structural)
    modified=copy.deepcopy(forecasts);modified[0]['prediction_id']='replacement'
    with pytest.raises(ValueError,match='population'):
        c.load_pre_race_field_policy(path,c.digest(path),modified,title,name,structural)
    doc=json.loads(path.read_text());doc['criteria']['test']='changed';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='criteria'):
        c.load_pre_race_field_policy(path,c.digest(path),forecasts,title,name,structural)


def test_v6_duplicate_archive_prediction_id_rejected(tmp_path):
    path,forecasts,title,name,structural=field_policy_fixture(tmp_path)
    source=tmp_path/'originals.json';rows=json.loads(source.read_text());rows.append(rows[0]);source.write_text(json.dumps(rows))
    doc=json.loads(path.read_text());doc['authenticated_originals_reference']['sha256']=c.digest(source);path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='membership'):
        c.load_pre_race_field_policy(path,c.digest(path),forecasts,title,name,structural)
