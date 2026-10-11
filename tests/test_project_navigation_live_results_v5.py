import pytest
from scripts import project_navigation_live_results_v5 as p
from tests.test_project_acquired_live_results import URL, original, markup, item
from tests.test_project_structured_live_results_v4 import body
from tests.test_project_policy_live_results_v3 import policy, POLICY_REF

NAV_REF = {'path':'/synthetic/navigation-policy.json','sha256':'3'*64}
NAME_REF = {'path':'/synthetic/runner-name-policy.json','sha256':'4'*64}


def navigation(raw=None):
    raw = raw or markup([(1,'Alpha','1st'),(2,'Beta','2nd')])
    raw=raw.replace(f'<link rel="canonical" href="{URL}">'.encode(),b'<title>thedogs - Sponsor Display 17 July 2026 Race 7</title>')
    relative=URL.replace('https://www.thedogs.com.au','')
    return raw + (f'<a class="race-box race-box--active race-box--result" href="{relative}"><div class="race-box__number">R7</div></a>'
                  f'<div class="page__layout"><div class="tabs-wrapper"><a class="tabs__item tabs__item--active" href="{relative}"><span>RESULTS</span></a></div></div>'
                  '<a class="dropdown__menu__meeting--active" href="/racing/healesville/2026-07-17?trial=false">Sponsor Display</a>').encode()


def test_exact_routes_establish_identity_without_display_alias(tmp_path):
    result=p.project_one(item(tmp_path,navigation()),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF)
    assert result['races'][0]['source_identity']['identity_semantics']=='exact_active_official_race_navigation_v5'
    assert result['races'][0]['source_identity']['active_race_number']==7
    assert 'runner_name_policy_fallback' not in result


@pytest.mark.parametrize('before,after',[
 ('race-box--active','race-box--inactive'),
 ('R7</div>','R8</div>'),
 ('>RESULTS<','>FORM<'),
 ('/racing/healesville/2026-07-17?trial=false','/racing/healesville/2026-07-18?trial=false'),
 ('Sponsor Display 17 July','Sponsor Display 18 July'),
])
def test_conflicting_or_missing_navigation_rejected(tmp_path,before,after):
    with pytest.raises(ValueError,match='source_navigation_'):
        p.project_one(item(tmp_path,navigation().replace(before.encode(),after.encode())),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF)


def test_name_repair_can_follow_exact_navigation_identity(tmp_path):
    trail={}
    result=p.project_one(item(tmp_path,navigation(body())),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF,NAME_REF,trail)
    assert [r['dog_name'] for r in result['runners']]==['Alpha','Beta']
    assert trail['status']=='QUALIFIED'
    assert result['races'][0]['structured_runner_name_witnesses'][0]['source_dog_id']=='123'


def test_true_name_change_or_unknown_terminal_stays_rejected(tmp_path):
    trail={}
    with pytest.raises(ValueError,match='terminal_status_missing:2'):
        p.project_one(item(tmp_path,navigation(body(status='F'))),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF,NAME_REF,trail)
    assert trail['status']=='REJECTED'


def test_old_and_new_structural_name_extractors_agree():
    from scripts import project_structured_live_results_v4 as old
    from bs4 import BeautifulSoup
    raw=body()
    rows,_,witnesses=old.parse_official_html(raw,URL,original(),policy(),POLICY_REF)
    native=p.parse_thedogs_result_html_runner_rows(raw.decode())
    fixed,actual=p.structured_name_rows(BeautifulSoup(raw.decode(),'html.parser'),native)
    assert fixed==rows and actual==witnesses


@pytest.mark.parametrize('extra',[
 b'<link rel="canonical" href="">',
 b'<meta property="og:url" content="https://www.thedogs.com.au/racing/healesville/2026-07-17/7/db-construction?trial=false">',
 b'<a class="race-box--active" href="/racing/healesville/2026-07-17/8/other?trial=false"></a>',
 b'<a class="dropdown__menu__meeting--active" href="/racing/healesville/2026-07-17?trial=false"></a>',
])
def test_ambiguous_or_declared_identity_cannot_use_navigation(tmp_path,extra):
    with pytest.raises(ValueError,match='source_navigation_'):
        p.project_one(item(tmp_path,navigation()+extra),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF)


@pytest.mark.parametrize('href',[
 '//www.thedogs.com.au/racing/healesville/2026-07-17/7/db-construction?trial=false',
 'https:/racing/healesville/2026-07-17/7/db-construction?trial=false',
 'https://foreign.example/racing/healesville/2026-07-17/7/db-construction?trial=false',
 '/racing/healesville/2026-07-17/7/db-construction?trial=false&extra=1',
 '/racing/healesville/2026-07-17/7/wrong-suffix?trial=false',
])
def test_route_not_exact_approved_href_rejected(tmp_path,href):
    relative=URL.replace('https://www.thedogs.com.au','')
    raw=navigation().replace(relative.encode(),href.encode())
    with pytest.raises(ValueError,match='source_navigation_'):
        p.project_one(item(tmp_path,raw),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF)


def test_known_other_venue_display_contradiction_rejected(tmp_path):
    mapping={'mappings':policy()['mappings']+[{'native_venue':'WRONG','official_url_slug':'wrong','proposed_exact_title_venues':['Sponsor Display']}]}
    with pytest.raises(ValueError,match='known_title_venue_contradiction'):
        p.project_one(item(tmp_path,navigation()),original(),URL,tmp_path/'derived',NAV_REF,mapping,POLICY_REF)


def test_global_active_site_tab_does_not_create_extra_result_tab(tmp_path):
    raw=navigation()+b'<a class="tabs__item tabs__item--active" href="/racing">RACING</a>'
    result=p.project_one(item(tmp_path,raw),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF)
    assert result['official_runner_identity_observed']


def test_integrator_preserves_prior_identity_failure_and_new_repair(tmp_path):
    from scripts import integrate_acquired_live_results as integration
    from tests.test_project_policy_live_results_v3 import ready
    obs=ready(item(tmp_path,navigation(body())))
    result=integration.project_observation(obs,original(),URL,tmp_path/'derived',policy(),POLICY_REF,{},NAME_REF,{},NAV_REF)
    assert result['projection_policy_fallback']['status']=='REJECTED'
    assert result['structural_identity_policy_fallback']['status']=='QUALIFIED'
    assert result['runner_name_policy_fallback']['status']=='QUALIFIED'
    assert result['structural_identity_policy_fallback']['original_parser_failure']==result['projection_policy_fallback']['fallback_failure']


def test_integrator_requires_external_structural_policy(tmp_path):
    from scripts import integrate_acquired_live_results as integration
    from tests.test_project_policy_live_results_v3 import ready
    with pytest.raises(ValueError,match='not_exact_policy_member'):
        integration.project_observation(ready(item(tmp_path,navigation())),original(),URL,tmp_path/'derived',policy(),POLICY_REF,{},NAME_REF)


def test_no_claim_of_name_repair_when_plain_names_already_validate(tmp_path):
    result=p.project_one(item(tmp_path,navigation()),original(),URL,tmp_path/'derived',NAV_REF,policy(),POLICY_REF,NAME_REF,{})
    assert result['response_provenance']['runner_name_policy_reference'] is None
    assert result['races'][0]['structured_runner_name_witnesses']==[]


def test_actual_structural_policy_population_and_dependency_authentication(tmp_path):
    import json
    from scripts import integrate_acquired_live_results as integration
    records=[{'race_id':f'Race {i}','prediction_id':str(i)} for i in range(210)]
    population=tmp_path/'population.json';population.write_text(json.dumps({'records':records}))
    review=tmp_path/'review.json';review.write_text('{}')
    doc={'schema':'frozen_offline_active_route_identity_policy_v5','status':'FROZEN_BEFORE_CORRECTED_METRICS',
         'field_and_scoring_policy_unchanged':True,'no_new_provider_or_runtime_authority':True,
         'population_reference':integration.reference(population),'source_title_policy_reference':POLICY_REF,
         'runner_name_policy_reference':NAME_REF,'independent_review_reference':integration.reference(review),
         'prior_parser_references':[integration.reference(m.__file__) for m in (integration.projector,integration.reviewed_projector,integration.policy_projector,integration.name_projector)]}
    path=tmp_path/'policy.json';path.write_text(json.dumps(doc));ref=integration.reference(path);allowed={r['race_id']:r['prediction_id'] for r in records}
    assert integration.verified_structural_identity_policy(ref,allowed,POLICY_REF,NAME_REF)==doc
    with pytest.raises(ValueError,match='population_mismatch'):
        integration.verified_structural_identity_policy(ref,{},POLICY_REF,NAME_REF)
    with pytest.raises(ValueError,match='dependency_policy_mismatch'):
        integration.verified_structural_identity_policy(ref,allowed,POLICY_REF,None)
    with pytest.raises(ValueError,match='hash'):
        integration.verified_structural_identity_policy({**ref,'sha256':'0'*64},allowed,POLICY_REF,NAME_REF)
