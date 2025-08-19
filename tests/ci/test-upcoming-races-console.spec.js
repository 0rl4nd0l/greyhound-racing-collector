/**
 * CI Test: Upcoming Races Console Error Validation
 * 
 * This test validates that the /upcoming_races endpoint loads without console errors.
 * It's designed to run in CI to prevent regressions on future commits.
 */

const { test, expect } = require('@playwright/test');

test.describe('Upcoming Races Console Error Validation', () => {
  let consoleErrors = [];
  let consoleWarnings = [];

  test.beforeEach(async ({ page }) => {
    consoleErrors = [];
    consoleWarnings = [];
    
    // Capture console messages
    page.on('console', (message) => {
      const type = message.type();
      const text = message.text();
      
      if (type === 'error') {
        consoleErrors.push(text);
        console.log(`❌ Console Error: ${text}`);
      } else if (type === 'warning') {
        consoleWarnings.push(text);
        console.log(`⚠️  Console Warning: ${text}`);
      }
    });
    
    // Capture page errors
    page.on('pageerror', (error) => {
      consoleErrors.push(`Page Error: ${error.message}`);
      console.log(`💥 Page Error: ${error.message}`);
    });
  });

  test('upcoming_races console errors - should have zero console errors', async ({ page }) => {
    console.log('🚀 Starting upcoming_races console error validation test...');
    
    // Navigate to the upcoming races page
    console.log('📍 Navigating to /upcoming...');
    const response = await page.goto('/upcoming', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Verify the page loaded successfully
    expect(response.status()).toBe(200);
    console.log('✅ Page loaded successfully with status 200');
    
    // Wait for page to be fully loaded
    await page.waitForLoadState('domcontentloaded');
    console.log('📄 DOM content loaded');
    
    // Wait a bit more for any async operations
    await page.waitForTimeout(3000);
    console.log('⏱️  Waited for async operations to complete');
    
    // Check for basic page elements to ensure it rendered correctly
    await expect(page.locator('body')).toBeVisible();
    console.log('👁️  Page body is visible');
    
    // Try to find some expected elements (non-blocking)
    const hasTitle = await page.locator('title, h1, h2').count() > 0;
    if (hasTitle) {
      console.log('📝 Page has title/heading elements');
    }
    
    // Log console message summary
    console.log(`📊 Console Summary:`);
    console.log(`   Errors: ${consoleErrors.length}`);
    console.log(`   Warnings: ${consoleWarnings.length}`);
    
    if (consoleErrors.length > 0) {
      console.log('❌ Console Errors found:');
      consoleErrors.forEach((error, index) => {
        console.log(`   ${index + 1}. ${error}`);
      });
    }
    
    if (consoleWarnings.length > 0) {
      console.log('⚠️  Console Warnings found:');
      consoleWarnings.forEach((warning, index) => {
        console.log(`   ${index + 1}. ${warning}`);
      });
    }
    
    // The main assertion: no console errors should be present
    expect(consoleErrors).toHaveLength(0);
    console.log('✅ SUCCESS: No console errors found on /upcoming page');
  });

  test('upcoming_races API endpoint - should respond without errors', async ({ page }) => {
    console.log('🔗 Testing /api/upcoming_races API endpoint...');
    
    // Test the API endpoint directly
    const apiResponse = await page.request.get('/api/upcoming_races');
    expect(apiResponse.status()).toBe(200);
    
    const apiData = await apiResponse.json();
    expect(apiData).toHaveProperty('success');
    expect(apiData.success).toBe(true);
    expect(apiData).toHaveProperty('races');
    expect(Array.isArray(apiData.races)).toBe(true);
    
    console.log(`✅ API endpoint returned ${apiData.races.length} races successfully`);
  });

  test('upcoming_races page JavaScript execution - should complete without errors', async ({ page }) => {
    console.log('⚡ Testing JavaScript execution on upcoming races page...');
    
    // Navigate to the page
    await page.goto('/upcoming', { waitUntil: 'networkidle' });
    
    // Execute some basic JavaScript to ensure the page is functional
    const pageTitle = await page.evaluate(() => {
      // Try to access common DOM elements
      return {
        title: document.title || 'No title',
        hasBody: !!document.body,
        hasHead: !!document.head,
        readyState: document.readyState,
        userAgent: navigator.userAgent.includes('Chrome') ? 'Chrome-based' : 'Other'
      };
    });
    
    console.log(`📋 Page Info: ${JSON.stringify(pageTitle, null, 2)}`);
    
    // Verify basic page structure exists
    expect(pageTitle.hasBody).toBe(true);
    expect(pageTitle.hasHead).toBe(true);
    expect(pageTitle.readyState).toBe('complete');
    
    // Ensure no console errors occurred during JavaScript execution
    expect(consoleErrors).toHaveLength(0);
    console.log('✅ JavaScript execution completed without console errors');
  });

  test.afterEach(async ({ page }) => {
    // Final check for any lingering console errors
    await page.waitForTimeout(1000);
    
    if (consoleErrors.length > 0) {
      console.log(`🔍 Final Error Count: ${consoleErrors.length}`);
      console.log('❌ Test completed with console errors - this will fail the CI build');
    } else {
      console.log('✅ Test completed successfully with zero console errors');
    }
  });
});
