import { test, expect } from '@playwright/test';

// This spec verifies that when DISABLE_NAV_DROPDOWNS=1 the app injects a
// style block that hides top navbar dropdowns. We validate by checking for the
// presence of the <style id="disable-nav-dropdowns"> tag and a couple of
// expected CSS selectors in the page HTML.

test.describe('Navbar dropdowns hidden when DISABLE_NAV_DROPDOWNS=1', () => {
  test('injects hide-css on home page', async ({ page }) => {
    await page.goto('/');
    // Use HTML content inspection for robustness across different head/body structures
    const html = await page.content();
    expect(html).toContain('id="disable-nav-dropdowns"');
    expect(html).toContain('.navbar .nav-item.dropdown');
    expect(html).toContain('.navbar .dropdown-menu');
    expect(html).toContain('.navbar .nav-link.dropdown-toggle');
  });

  test('injects hide-css on test-blank-page as well', async ({ page }) => {
    await page.goto('/test-blank-page');
    const html = await page.content();
    expect(html).toContain('id="disable-nav-dropdowns"');
    expect(html).toContain('.navbar .nav-item.dropdown');
  });
});
