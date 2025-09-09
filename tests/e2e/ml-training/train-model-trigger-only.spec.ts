import { test, expect } from "@playwright/test";

test.describe("ML Training Page - trigger only", () => {
  test("start training via UI button click", async ({ page }) => {
    test.setTimeout(60000);
    await page.goto("/ml-training");

    // Click the registry-backed trigger button
    const btn = page.getByRole("button", { name: /Train New Model/i });
    await expect(btn).toHaveCount(1);

    const [resp] = await Promise.all([
      page.waitForResponse((r) => r.url().includes("/api/model/training/trigger") && r.request().method() === "POST" && r.ok(), { timeout: 20000 }),
      btn.click(),
    ]);

    // Confirm we got OK from trigger endpoint
    const body = await resp.json();
    expect(body && body.success).toBeTruthy();
    // Do not wait for progress UI here; job continues server-side
  });
});

