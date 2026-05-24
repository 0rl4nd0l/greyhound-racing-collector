import { test, expect } from "@playwright/test";

// End-to-end UI validation that starts a training job and attaches to the latest running job
// so the Training Progress & Logs panel shows live updates.

test.describe("ML Training Page - start + attach", () => {
  test("Start registry training and attach to latest running job", async ({ page }) => {
    test.setTimeout(90000);
    await page.goto("/ml-training");

    // Click the registry-backed trigger and confirm the POST succeeds
    const triggerBtn = page.getByRole("button", { name: /Train New Model/i });
    await expect(triggerBtn).toHaveCount(1);
    const [resp] = await Promise.all([
      page.waitForResponse((r) => r.url().includes("/api/model/training/trigger") && r.request().method() === "POST" && r.ok(), { timeout: 20000 }),
      triggerBtn.click(),
    ]);
    const body = await resp.json();
    expect(body && body.success).toBeTruthy();

    // Progress bar should become visible after trigger
    const progressBar = page.locator("#training-progress-bar");
    // If not visible in ~5s, click Attach to Job to bind the panel
    try {
      await expect(progressBar).toBeVisible({ timeout: 5000 });
    } catch {
      const attachBtn = page.getByRole("button", { name: /Attach to Job/i });
      await attachBtn.click();
      await expect(progressBar).toBeVisible({ timeout: 10000 });
    }

    // Status should update or show spinner text
    const statusEl = page.locator("#training-status");
    await expect(statusEl).toContainText(/Status:|Training|spinner/i, { timeout: 15000 });

    // Logs should start appending lines
    const logs = page.locator("#training-logs div");
    await expect(logs.first()).toBeVisible({ timeout: 20000 });
  });
});

