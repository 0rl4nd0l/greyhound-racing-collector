import { test, expect, Page } from '@playwright/test';

async function waitForMenu(page: Page) {
  // Wait for the global flag and the menu container or toolbar to appear
  await page.waitForFunction(() => (window as any).ENDPOINTS_MENU_ENABLED === true, undefined, { timeout: 10000 });
  await page.waitForFunction(
    () => !!document.querySelector('.endpoints-menu') || !!document.getElementById('endpoints-menu-toolbar'),
    undefined,
    { timeout: 10000 }
  );
}

async function selectOptionByValue(page: Page, value: string) {
  const select = page.locator(`select:has(option[value="${value}"])`).first();
  await expect(select).toBeVisible();
  await select.selectOption(value);
}

test.describe('Endpoints Menu - dynamic dropdowns', () => {
  test('renders endpoints toolbar and lists routes', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });

    await waitForMenu(page);

    // Ensure the menu container exists
    await page.waitForFunction(
      () => !!document.querySelector('.endpoints-menu') || !!document.getElementById('endpoints-menu-toolbar')
    );

    // Verify the endpoints API returns data
    const apiResp = await page.request.get('/api/endpoints');
    expect(apiResp.status()).toBe(200);
    const apiData = await apiResp.json();
    expect(apiData).toHaveProperty('success');
    expect(apiData.success).toBe(true);
    expect(Array.isArray(apiData.endpoints)).toBe(true);
    expect(apiData.endpoints.length).toBeGreaterThan(0);

    // There should be at least one GET select
    const selectsCount = await page.locator('.endpoints-menu select').count();
    expect(selectsCount).toBeGreaterThan(0);
  });

  test('can invoke GET and POST endpoints from the menu', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    await waitForMenu(page);

    // GET: /api/model/performance
    const perfResponsePromise = page.waitForResponse((r) => r.url().includes('/api/model/performance'));
    await selectOptionByValue(page, '/api/model/performance');
    const perfResp = await perfResponsePromise;
    expect(perfResp.status()).toBe(200);

    // POST: /api/model/monitoring/drift (respond to prompt with JSON)
    const driftResponsePromise = page.waitForResponse((r) => r.url().includes('/api/model/monitoring/drift'));
    page.once('dialog', async (dialog) => {
      await dialog.accept('{"window": 10}');
    });
    await selectOptionByValue(page, '/api/model/monitoring/drift');
    const driftResp = await driftResponsePromise;
    expect(driftResp.status()).toBe(200);

    // Optionally, verify response bodies are JSON
    const driftJson = await driftResp.json().catch(() => null);
    expect(driftJson).not.toBeNull();
    expect(driftJson.success).toBeTruthy();
  });
});
