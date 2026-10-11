import pytest
from scripts import project_structured_live_results_v4 as p
from scripts import integrate_acquired_live_results as integration
from tests.test_project_acquired_live_results import URL, item, markup, original
from tests.test_project_policy_live_results_v3 import ready, policy, POLICY_REF

NAME_REF = {'path': '/synthetic/name-policy.json', 'sha256': '1'*64}


def structured(name, dog_id=123, badge='NBT', box_note=''):
    return (f'<blackbook-dog data-dog-id="{dog_id}"></blackbook-dog>'
            f'<div class="race-runners__name__dog"><a href="/dogs/{dog_id}/native-name">{name}</a>'
            f'<span class="race-runners__name__time">{badge}</span>'
            f'<span class="race-runners__name__box">{box_note}</span></div>'
            '<div class="race-runners__name__trainer">T: Trainer</div>')


def body(beta='Beta', status='2nd'):
    return markup([(1, structured('Alpha', 123), '1st'), (2, structured(beta, 456), status)])


def test_nbt_structural_repair_canonical_needs_no_title(tmp_path):
    obs = ready(item(tmp_path, body()))
    result = integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF, {}, NAME_REF)
    assert [r['dog_name'] for r in result['runners']] == ['Alpha', 'Beta']
    assert result['runner_name_policy_fallback']['original_parser_failure'] == 'official_runner_name_changed_at_box:1'
    assert result['runner_name_policy_fallback']['status'] == 'QUALIFIED'
    witness = result['races'][0]['structured_runner_name_witnesses'][0]
    assert witness['source_dog_id'] == '123'
    assert witness['separate_time_annotations'] == ['NBT']
    assert witness['original_flattened_name'] == 'Alpha NBT T: Trainer'
    assert witness['original_parser_name'] == 'Alpha NBT'


def test_actual_dog_name_ending_nbt_is_not_suffix_stripped(tmp_path):
    src = original()
    src['predictions'][1]['dog_name'] = 'Beta NBT'
    result = p.project_one(item(tmp_path, body('Beta NBT')), src, URL, tmp_path/'derived', policy(), POLICY_REF, NAME_REF)
    assert result['runners'][1]['dog_name'] == 'Beta NBT'


@pytest.mark.parametrize('status', ['-', 'F', 'T'])
def test_no_unknown_terminal_status_inference_after_name_repair(tmp_path, status):
    obs = ready(item(tmp_path, body(status=status)))
    with pytest.raises(ValueError, match='terminal_status_missing:2'):
        integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF, {}, NAME_REF)
    assert obs['runner_name_policy_fallback']['status'] == 'REJECTED'
    assert (tmp_path/'derived/structured-name-v4/parsed_official_response.json').exists()


@pytest.mark.parametrize('replace', [
    ('/dogs/123/native-name', 'https://foreign.example/dogs/123/native-name'),
    ('/dogs/123/native-name', 'https:/dogs/123/native-name'),
    ('/dogs/123/native-name', '//www.thedogs.com.au/dogs/123/native-name'),
    ('data-dog-id="123"', 'data-dog-id="999"'),
    ('>Alpha</a>', '>Alpha</a><a href="/dogs/124/other">Other</a>'),
])
def test_foreign_ambiguous_or_inconsistent_profile_rejected(tmp_path, replace):
    raw = body().replace(replace[0].encode(), replace[1].encode())
    with pytest.raises(ValueError, match='structured_dog_'):
        p.project_one(item(tmp_path, raw), original(), URL, tmp_path/'derived', policy(), POLICY_REF, NAME_REF)


def test_true_changed_name_still_rejected(tmp_path):
    with pytest.raises(ValueError, match='name_changed_at_box:2'):
        p.project_one(item(tmp_path, body('Replacement')), original(), URL, tmp_path/'derived', policy(), POLICY_REF, NAME_REF)


def test_title_fallback_and_name_repair_preserve_both_failure_trails(tmp_path):
    raw = body().replace(f'<link rel="canonical" href="{URL}">'.encode(), b'<title>thedogs - Healesville 17 July 2026 Race 7</title>')
    obs = ready(item(tmp_path, raw))
    result = integration.project_observation(obs, original(), URL, tmp_path/'derived', policy(), POLICY_REF, {}, NAME_REF)
    assert result['projection_policy_fallback']['status'] == 'REJECTED'
    assert result['runner_name_policy_fallback']['status'] == 'QUALIFIED'
    assert result['races'][0]['source_identity']['identity_semantics'] == 'frozen_exact_source_title_venue_date_race_v3'


def test_canonical_contradiction_cannot_be_repaired_as_name(tmp_path):
    raw = body() + b'<meta property="og:url" content="">'
    with pytest.raises(ValueError, match='source_declared'):
        p.project_one(item(tmp_path, raw), original(), URL, tmp_path/'derived', policy(), POLICY_REF, NAME_REF)


def test_missing_policy_leaves_original_failure(tmp_path):
    with pytest.raises(ValueError, match='name_changed_at_box:1'):
        integration.project_observation(ready(item(tmp_path, body())), original(), URL, tmp_path/'derived', policy(), POLICY_REF)


def test_unknown_terminal_after_repair_does_not_suppress_old_diagnostic(tmp_path):
    obs = ready(item(tmp_path, body(status='F')))
    obs['receipt_reference'] = {'path': '/synthetic/completion.json', 'sha256': '2'*64}
    prior = {'race_id': obs['race_id']}
    selected, dispositions, _ = integration.integrate([obs['race_id']], [prior], [obs],
        lambda row: integration.project_observation(row, original(), URL, tmp_path/'derived', policy(), POLICY_REF, {}, NAME_REF))
    assert selected == [prior]
    assert dispositions[0]['disposition'] == 'PRIOR_DIAGNOSTIC_RETAINED_AFTER_UNQUALIFIED_RESPONSE'
    assert dispositions[0]['reasons'] == ['official_runner_terminal_status_missing:2']


def test_promotion_box_note_preserved_in_native_rows(tmp_path):
    raw = markup([(1, structured('Alpha', 123), '1st'), (9, structured('Beta', 456, box_note='(from box 2)'), '2nd')])
    rows, _, witnesses = p.parse_official_html(raw, URL, original(), policy(), POLICY_REF)
    assert rows[1]['dog_name'] == 'Beta (from box 2)'
    assert rows[1]['box_number'] == 9
    assert witnesses[1]['source_box_note'] == '(from box 2)'


def test_policy_authentication_exact_population_and_parser_bytes(tmp_path):
    import json
    population = {'records': [{'race_id': f'Race {i}', 'prediction_id': str(i)} for i in range(210)]}
    pop_path = tmp_path/'population.json'
    pop_path.write_text(json.dumps(population))
    review_path = tmp_path/'review.json'
    review_path.write_text('{}')
    doc = {'schema': 'frozen_offline_structured_dog_name_policy_v4', 'status': 'FROZEN_BEFORE_CORRECTED_METRICS',
           'field_and_scoring_policy_unchanged': True, 'no_new_provider_or_runtime_authority': True,
           'population_reference': integration.reference(pop_path), 'source_title_policy_reference': POLICY_REF,
           'prior_parser_references': [integration.reference(m.__file__) for m in (integration.projector,integration.reviewed_projector,integration.policy_projector)],
           'independent_review_reference': integration.reference(review_path)}
    path = tmp_path/'policy.json'
    path.write_text(json.dumps(doc))
    allowed = {r['race_id']: r['prediction_id'] for r in population['records']}
    assert integration.verified_runner_name_policy(integration.reference(path), allowed, POLICY_REF) == doc
    with pytest.raises(ValueError, match='hash'):
        integration.verified_runner_name_policy({'path':str(path), 'sha256':'0'*64}, allowed, POLICY_REF)
    with pytest.raises(ValueError, match='population_mismatch'):
        integration.verified_runner_name_policy(integration.reference(path), {}, POLICY_REF)
    with pytest.raises(ValueError, match='title_policy_mismatch'):
        integration.verified_runner_name_policy(integration.reference(path), allowed, None)


@pytest.mark.parametrize('mutation', [
    lambda raw: raw.replace(b'</a><span class="race-runners__name__time">', b'</a><span class="race-runners__name__time">duplicate</span><span class="race-runners__name__time">', 1),
    lambda raw: raw.replace(b'<sprite-svg name="rug_1"></sprite-svg>', b'<sprite-svg name="rug_1"></sprite-svg><sprite-svg name="rug_1"></sprite-svg>'),
    lambda raw: raw.replace(b'/dogs/123/native-name', b'/dogs/0/native-name'),
])
def test_ambiguous_badge_box_and_invalid_id_rejected(tmp_path, mutation):
    with pytest.raises(ValueError, match='structured_dog_'):
        p.project_one(item(tmp_path, mutation(body())), original(), URL, tmp_path/'derived', policy(), POLICY_REF, NAME_REF)
