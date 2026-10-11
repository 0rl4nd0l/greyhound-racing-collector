import json
import pytest
from scripts import project_nontrial_navigation_live_results_v7 as p
from scripts import integrate_acquired_live_results as integration
from tests.test_project_acquired_live_results import URL,item,original
from tests.test_project_policy_live_results_v3 import policy,POLICY_REF,ready
from tests.test_project_structured_live_results_v4 import body,NAME_REF
from tests.test_project_navigation_live_results_v5 import navigation,NAV_REF
from tests.test_project_prerace_field_live_results_v6 import wrapper,resign_sidecar,FIELD_REF
V7_REF={'path':'/synthetic/nontrial-policy.json','sha256':'7'*64}
BARE=URL.split('?')[0]


def raw(status='2nd'):
    return navigation(body(status=status)).replace(b'tabs__item tabs__item--active',b'tabs__item')


def inputs(tmp_path,html=None,expected=BARE):
    data=wrapper(tmp_path)
    def mutate(value):
        value['canonical_runner_alignment']['canonical_source_url']=expected
        value['prejump_shadow_metadata']['source_url']=expected
    resign_sidecar(data,mutate)
    request=item(tmp_path,html or raw())
    from pathlib import Path
    meta=Path(request['metadata_path']);value=json.loads(meta.read_text());value.update(requested_url=expected,final_url=expected);meta.write_text(json.dumps(value))
    return request,data


def project(tmp_path,html=None,expected=BARE):
    request,data=inputs(tmp_path,html,expected)
    return p.project_one(request,data,expected,tmp_path/'derived',NAV_REF,policy(),POLICY_REF,V7_REF,NAME_REF,{})


@pytest.mark.parametrize('expected',[BARE,URL])
def test_exact_source_nontrial_navigation_without_active_tab(tmp_path,expected):
    result=project(tmp_path,expected=expected)
    identity=result['races'][0]['source_identity']
    assert identity['identity_semantics']=='exact_source_nontrial_navigation_v7'
    assert identity['results_anchor_classes']==['tabs__item']
    assert identity['results_href'].endswith('?trial=false')
    assert identity['pre_race_identity_proof']['sidecar_reference']['sha256']
    assert [r['dog_name'] for r in result['runners']]==['Alpha','Beta']


@pytest.mark.parametrize('query',['','?trial=true','?trial=','?trial=false&trial=false','?trial=false&extra=1','?trial=%66alse','#results'])
def test_source_query_must_be_explicit_literal_false(tmp_path,query):
    html=raw().replace(b'/7/db-construction?trial=false',('/7/db-construction'+query).encode())
    with pytest.raises(ValueError,match='source_navigation_href_'):
        project(tmp_path,html)


@pytest.mark.parametrize('suffix',['?trial=true','?trial=','?trial=false&x=1','?trial=false&trial=false','?trial=%66alse','#fragment','?'])
def test_authorized_query_cannot_expand_supported_scope(tmp_path,suffix):
    with pytest.raises(ValueError,match='official_url_trial_scope|source_nontrial_authorized_url'):
        project(tmp_path,expected=BARE+suffix)


@pytest.mark.parametrize('extra',[
 b'<link rel="canonical" href="">',
 b'<meta property="og:url" content="https://www.thedogs.com.au/racing/healesville/2026-07-17/7/db-construction">',
 b'<div class="page__layout"><div class="tabs-wrapper"><a class="tabs__item" href="/racing/other">RESULTS</a></div></div>',
 b'<div class="page__layout"><div class="tabs-wrapper"><a class="tabs__item tabs__item--active" href="/racing/other">FORM</a></div></div>',
])
def test_declaration_or_conflicting_result_navigation_rejected(tmp_path,extra):
    with pytest.raises(ValueError,match='source_navigation_declared|source_nontrial_results|source_nontrial_active_tab'):
        project(tmp_path,raw()+extra)


@pytest.mark.parametrize('status',['-','F','T','SCR'])
def test_identity_repair_never_invents_terminal_status_or_drops_forecast_runner(tmp_path,status):
    with pytest.raises(ValueError,match='terminal_status_missing|forecast_runner_scratched'):
        project(tmp_path,raw(status))


def test_known_display_with_delimiter_whitespace_cannot_bypass_contradiction():
    other={'mappings':policy()['mappings']+[{'native_venue':'WRONG','official_url_slug':'wrong','proposed_exact_title_venues':['Sponsor Display']}]}
    html=raw().replace(b'Sponsor Display 17 July',b'Sponsor Display  17 July')
    with pytest.raises(ValueError,match='known_title_venue_contradiction'):
        p.parse_official_html(html,BARE,original(),other,NAV_REF,POLICY_REF,V7_REF)


def test_preseal_sidecar_must_match_authorized_url(tmp_path):
    request,data=inputs(tmp_path);resign_sidecar(data,lambda value:value['canonical_runner_alignment'].update(canonical_source_url=URL.replace('/7/','/8/')))
    with pytest.raises(ValueError,match='pre_race_field_source_url_mismatch'):
        p.project_one(request,data,BARE,tmp_path/'derived',NAV_REF,policy(),POLICY_REF,V7_REF,NAME_REF,{})


def test_integrator_preserves_prior_failures_and_source_hold_rules(tmp_path):
    request,data=inputs(tmp_path);obs=ready(request)
    result=integration.project_observation(obs,data['original'],BARE,tmp_path/'derived',policy(),POLICY_REF,{},NAME_REF,{},NAV_REF,{},FIELD_REF,data,{},V7_REF)
    assert result['projection_policy_fallback']['status']=='REJECTED'
    assert result['structural_identity_policy_fallback']['status']=='REJECTED'
    assert result['nontrial_navigation_policy_fallback']['status']=='QUALIFIED'
    assert result['runner_name_policy_fallback']['status']=='QUALIFIED'
    assert result['nontrial_navigation_policy_fallback']['original_parser_failure']=='source_navigation_href_not_exact_approved_route'
    obs=ready(request);obs['source_hold']=True
    with pytest.raises(ValueError,match='ready_unheld'):
        integration.project_observation(obs,data['original'],BARE,tmp_path/'held',policy(),POLICY_REF,{},NAME_REF,{},NAV_REF,{},FIELD_REF,data,{},V7_REF)
    assert 'nontrial_navigation_policy_fallback' not in obs


def test_absent_v7_policy_keeps_original_refusal(tmp_path):
    request,data=inputs(tmp_path)
    with pytest.raises(ValueError,match='source_navigation_href_not_exact_approved_route'):
        integration.project_observation(ready(request),data['original'],BARE,tmp_path/'derived',policy(),POLICY_REF,{},NAME_REF,{},NAV_REF,{},FIELD_REF,data)


def test_v7_does_not_silently_compose_an_inactive_field_repair(tmp_path):
    from tests.test_project_prerace_field_live_results_v6 import official_body
    html=navigation(official_body()).replace(b'tabs__item tabs__item--active',b'tabs__item')
    # The sealed proof contains this inactive identity; V7 only repairs identity.
    # A future combined V7+V6 field transform requires an explicit tested seam.
    with pytest.raises(ValueError,match='official_runner_outside_forecast_field'):
        project(tmp_path,html)
