import pytest
from scripts import project_reviewed_live_results_v2 as p
from tests.test_project_acquired_live_results import URL, item, markup, original


def titled(rows=None, title='thedogs - Healesville 17 July 2026 Race 7', extra=b''):
    raw = markup(rows or [(1, 'Alpha', '1st'), (2, 'Beta', '2nd')])
    raw = raw[raw.index(b'<table'):]
    return f'<title>{title}</title>'.encode() + extra + raw


def test_exact_title_and_native_field_without_canonical(tmp_path):
    raw = titled(extra=b'<script>src="https://www.recaptcha.net/recaptcha/api.js" async defer</script>')
    result = p.project_one(item(tmp_path, raw), original(), URL, tmp_path/'derived')
    identity = result['races'][0]['source_identity']
    assert identity['source_title_identity'] == ['healesville', '2026-07-17', 7]
    assert identity['reviewed_ordinary_recaptcha_script_count'] == 1
    assert result['official_runner_identity_observed'] is True


@pytest.mark.parametrize('title', [
    'thedogs - Healesville 18 July 2026 Race 7',
    'thedogs - Healesville 17 July 2026 Race 8',
    'thedogs - Healesville 31 February 2026 Race 7',
    'thedogs - HEA 17 July 2026 Race 7',
    'thedogs - healesville 17 July 2026 Race 7',
    'thedogs - Healesville 17 July 2026 Race 7 extra',
])
def test_title_identity_never_fuzzy_or_calendar_invalid(tmp_path, title):
    with pytest.raises(ValueError):
        p.project_one(item(tmp_path, titled(title=title)), original(), URL, tmp_path/'derived')


@pytest.mark.parametrize('extra', [
    b'<title>thedogs - Healesville 17 July 2026 Race 7</title>',
    b'<link rel="canonical" href="">',
    b'<meta property="og:url">',
    b'<link rel="canonical" href="bad">',
    f'<meta property="og:url" content="{URL.replace("/7/", "/8/")}">'.encode(),
])
def test_title_cannot_override_other_ambiguous_or_contradictory_identity(tmp_path, extra):
    with pytest.raises(ValueError):
        p.project_one(item(tmp_path, titled(extra=extra)), original(), URL, tmp_path/'derived')


@pytest.mark.parametrize('extra', [
    b'<div>Please complete captcha</div>',
    b'<script>var captcha = true;</script>',
    b'<script src="https://other.test/captcha"></script>',
    b'<div class="cf-chl-widget"></div>',
    b'<script>var turnstile = true;</script>',
    b'<h1>Verify you are human</h1>',
])
def test_false_positive_exception_does_not_accept_other_challenge_contexts(tmp_path, extra):
    with pytest.raises(ValueError, match='unapproved_challenge_context'):
        p.project_one(item(tmp_path, titled(extra=extra)), original(), URL, tmp_path/'derived')


@pytest.mark.parametrize('rows,reason', [
    ([(1, 'Alpha', '1st'), (2, 'Changed', '2nd')], 'name_changed'),
    ([(1, 'Alpha', '1st'), (2, 'Beta', 'SCR')], 'runner_scratched'),
    ([(1, 'Alpha', '1st'), (2, 'Beta', '1st')], 'win_target_ambiguous'),
])
def test_title_adapter_preserves_original_field_gates(tmp_path, rows, reason):
    with pytest.raises(ValueError, match=reason):
        p.project_one(item(tmp_path, titled(rows)), original(), URL, tmp_path/'derived')
