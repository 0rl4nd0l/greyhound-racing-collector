import copy
import json
from pathlib import Path
import pytest
from scripts import project_policy_live_results_v3 as p
from scripts import integrate_acquired_live_results as integration
from tests.test_project_acquired_live_results import URL, RID, item, markup, original
from tests.test_project_reviewed_live_results_v2 import titled

POLICY_REF = {'path': '/synthetic/frozen-policy.json', 'sha256': '0'*64}


def policy():
    return {'mappings': [{'native_venue': 'HEA', 'official_url_slug': 'healesville',
                          'proposed_exact_title_venues': ['Healesville']}]}


def ready(request):
    return {'race_id': RID, 'status': integration.READY, 'source_hold': False, 'projection_item': request}


def test_ready_missing_canonical_falls_back_with_original_failure_and_policy(tmp_path):
    obs = ready(item(tmp_path, titled()))
    result = integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)
    assert result['projection_policy_fallback']['original_parser_failure'] == 'source_declared_race_identity_missing_or_mismatched'
    assert result['projection_policy_fallback']['status'] == 'QUALIFIED'
    assert result['races'][0]['source_identity']['projection_policy_reference'] == POLICY_REF
    assert result['response_provenance']['reused_base_projector_sha256']


def test_valid_canonical_ready_never_requires_title(tmp_path):
    obs = ready(item(tmp_path, markup([(1, 'Alpha', '1st'), (2, 'Beta', 'DNF')])))
    result = integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)
    assert 'projection_policy_fallback' not in result
    assert result['runners'][1]['terminal_status'] == 'DNF'


@pytest.mark.parametrize('extra', [
    b'<link rel="canonical" href="">', b'<meta property="og:url">',
    f'<link rel="canonical" href="{URL.replace("/7/", "/8/")}">'.encode(),
    f'<meta property="og:url" content="{URL.replace("trial=false", "trial=true")}">'.encode(),
    b'<link rel="canonical" href="malformed">',
])
def test_fallback_never_overrides_declared_identity_contradictions(tmp_path, extra):
    obs = ready(item(tmp_path, titled(extra=extra)))
    with pytest.raises(ValueError):
        integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)


@pytest.mark.parametrize('title', [
    'thedogs - HEA 17 July 2026 Race 7',
    'thedogs - Healesville 18 July 2026 Race 7',
    'thedogs - Healesville 17 July 2026 Race 8',
    'thedogs - Healesville 31 February 2026 Race 7',
    'thedogs - Healesville 17 July 2026 Race 7 extra',
])
def test_title_policy_rejects_unlisted_or_mismatched_identity(tmp_path, title):
    obs = ready(item(tmp_path, titled(title=title)))
    with pytest.raises(ValueError):
        integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)
    assert obs['projection_policy_fallback']['status'] == 'REJECTED'


def test_canonical_field_failure_never_invokes_title_fallback(tmp_path):
    obs = ready(item(tmp_path, markup([(1, 'Alpha', '1st'), (2, 'Beta', 'SCR')])))
    with pytest.raises(ValueError, match='runner_scratched'):
        integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)
    assert 'projection_policy_fallback' not in obs


def test_source_hold_cannot_use_ordinary_title_policy(tmp_path):
    obs = ready(item(tmp_path, titled()))
    obs.update(status='TERMINAL_SOURCE_HOLD_NO_RETRY', source_hold=True)
    with pytest.raises(ValueError, match='original_ready_unheld'):
        integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)
    assert 'projection_policy_fallback' not in obs


def test_held_reviewed_path_remains_v2_even_with_policy(tmp_path):
    obs = ready(item(tmp_path, titled()))
    obs.update(status='TERMINAL_SOURCE_HOLD_NO_RETRY', source_hold=True, reviewed_reclassification={'synthetic': True})
    result = integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF)
    assert result['races'][0]['source_identity']['identity_semantics'] == 'exact_source_title_venue_date_race_v1'
    assert 'projection_policy_fallback' not in result


@pytest.mark.parametrize('display', ['Sandown', 'Sandown Park'])
def test_exact_preapproved_display_aliases_same_slug_native_identity(tmp_path, display):
    src = original()
    src['race_id'] = RID.replace('HEA', 'SAN')
    request = item(tmp_path, titled(title=f'thedogs - {display} 17 July 2026 Race 7'))
    expected = URL.replace('healesville', 'sandown')
    metadata = json.loads(Path(request['metadata_path']).read_text())
    metadata.update(requested_url=expected, final_url=expected)
    Path(request['metadata_path']).write_text(json.dumps(metadata))
    finite = {'mappings': [{'native_venue': 'SAN', 'official_url_slug': 'sandown', 'proposed_exact_title_venues': ['Sandown', 'Sandown Park']}]}
    result = p.project_one(request, src, expected, tmp_path/'derived', finite, POLICY_REF)
    assert result['races'][0]['source_identity']['source_title_native_venue'] == 'SAN'


def test_title_matching_another_native_venue_is_not_enough(tmp_path):
    finite = copy.deepcopy(policy())
    finite['mappings'][0]['native_venue'] = 'OTHER'
    with pytest.raises(ValueError, match='not_exact_policy_member'):
        p.project_one(item(tmp_path, titled()), original(), URL, tmp_path/'derived', finite, POLICY_REF)
