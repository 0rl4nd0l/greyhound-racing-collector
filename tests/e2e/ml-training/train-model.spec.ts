import { test, expect } from "@playwright/test";
import path from "path";

// This spec assumes the ML Training UI provides a modal or button
// that triggers POST /api/model/training/trigger and then polls
// GET /api/model/registry/status?job_id=...

async function startTraining(page) {
  // Open the training page
  await page.goto("/ml-training");

  // If there's a "Train New Model" button or similar, click it
  // Adjust selector if your page uses a different control
  const trainBtn = page.locator("button:has-text('Train New Model')");
  if (await trainBtn.count()) {
    await trainBtn.click();
  }

  // In case a modal appears with options, submit the form
  const startBtn = page.locator("button:has-text('Start')");
  if (await startBtn.count()) {
    await startBtn.click();
  }
}

async function waitForTrainingCompletion(page) {
  // Poll the registry endpoint for any active job then follow it to completion
  // Fallback polling window: up to 3 minutes
  const deadline = Date.now() + 3 * 60 * 1000;
  let jobId = null as string | null;

  while (Date.now() < deadline) {
    const res = await page.request.get("/api/model/registry/status");
    expect(res.ok()).toBeTruthy();
    const body = await res.json();

    const jobs = body.training_jobs || {};
    const running = Object.values(jobs).find((j: any) => j.status === "running" || j.status === "starting");
    if (running) {
      jobId = (running as any).id;
      break;
    }

    // If no running job found yet, give the UI a moment to spawn it
    await page.waitForTimeout(1000);
  }

  expect(jobId, "Expected a training job to start").toBeTruthy();

  // Follow the specific job until it completes
  while (Date.now() < deadline) {
    const res = await page.request.get(`/api/model/registry/status?job_id=${jobId}`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();

    if (body.status === "completed") {
      return;
    }
    if (body.status === "failed") {
      throw new Error(`Training failed: ${body.error_message || "unknown error"}`);
    }

    await page.waitForTimeout(1500);
  }

  throw new Error("Training did not complete within the expected time window");
}

async function generateData() {
  // Run the generator via a node child_process or expect pre-generated files.
  // In CI on this repo, the files will be generated before this test or we can
  // require the developer to run the generator once.
}

test.describe("ML Training Page", () => {
  test("clicking training triggers background job and completes", async ({ page }) => {
    // Ensure synthetic data exists (developer can run the generator separately if needed)
    // For now, we only validate that the UI wiring and backend job execute.

    await startTraining(page);
    await waitForTrainingCompletion(page);

    // Basic sanity: the models list should be accessible without error
    const res = await page.request.get("/api/model/registry/status");
    expect(res.ok()).toBeTruthy();
    const body = await res.json();

    // We at least expect a success flag; the rest depends on current registry state
    expect(body.success).toBeTruthy();
  });
});

