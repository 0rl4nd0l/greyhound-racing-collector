import app as app_module


def test_disable_nav_dropdowns_injects_css(client):
    """When DISABLE_NAV_DROPDOWNS is enabled, the app should inject a CSS block
    that hides navbar dropdowns. This checks for the presence of the style tag
    and expected selectors in the HTML response for a basic page.
    """
    # Force the feature flag at module scope (the app reads this at runtime in after_request)
    app_module.DISABLE_NAV_DROPDOWNS = True

    resp = client.get("/")
    assert resp.status_code == 200

    html = resp.data.decode("utf-8", errors="ignore")

    # The style tag should be injected
    assert (
        '<style id="disable-nav-dropdowns">' in html
    ), "Expected DISABLE_NAV_DROPDOWNS CSS to be injected into HTML head/body"

    # Basic sanity: selectors that should be hidden when the flag is set
    assert ".navbar .nav-item.dropdown" in html
    assert ".navbar .dropdown-menu" in html
    assert ".navbar .nav-link.dropdown-toggle" in html
