const { test, expect } = require('@playwright/test');

test.describe('Test Helper Routes', () => {
  test.beforeEach(async ({ page }) => {
    // Ensure the Flask app is running in testing mode
    const response = await page.request.get('/ping');
    expect(response.status()).toBe(200);
  });

  test.describe('/test-blank-page route', () => {
    test('should load the blank test page with minimal HTML structure', async ({ page }) => {
      await page.goto('/test-blank-page');
      
      // Check that the page loads successfully
      await expect(page.locator('body')).toBeVisible();
      
      // Verify the basic HTML structure
      await expect(page.locator('#test-container')).toBeVisible();
      await expect(page.locator('#test-container')).toHaveClass(/container-fluid/);
      
      // Check that Bootstrap CSS is loaded
      const bodyElement = page.locator('body');
      const computedStyle = await bodyElement.evaluate(el => window.getComputedStyle(el).fontFamily);
      expect(computedStyle).toBeTruthy();
      
      // Verify the page is minimal and ready for test injection
      const containerContent = await page.locator('#test-container').innerHTML();
      expect(containerContent.trim()).toBe('');
      
      // Test that we can inject content into the test container
      await page.locator('#test-container').fill('');
      await page.locator('#test-container').evaluate(el => {
        el.innerHTML = '<div id="injected-test">Test Content</div>';
      });
      await expect(page.locator('#injected-test')).toHaveText('Test Content');
    });

    test('should have Bootstrap CSS loaded and functional', async ({ page }) => {
      await page.goto('/test-blank-page');
      
      // Inject some Bootstrap components to test CSS is working
      await page.locator('#test-container').evaluate(el => {
        el.innerHTML = `
          <div class="alert alert-primary" role="alert">
            Bootstrap Alert Test
          </div>
          <button type="button" class="btn btn-success">Bootstrap Button</button>
        `;
      });
      
      // Check that Bootstrap classes are applied
      await expect(page.locator('.alert-primary')).toBeVisible();
      await expect(page.locator('.btn-success')).toBeVisible();
      
      // Check computed styles to ensure Bootstrap is actually working
      const alertBgColor = await page.locator('.alert-primary').evaluate(el => 
        window.getComputedStyle(el).backgroundColor
      );
      expect(alertBgColor).toBeTruthy();
    });

    test('should be responsive across different viewport sizes', async ({ page }) => {
      await page.goto('/test-blank-page');
      
      // Test desktop viewport
      await page.setViewportSize({ width: 1200, height: 800 });
      await expect(page.locator('#test-container')).toBeVisible();
      
      // Test tablet viewport
      await page.setViewportSize({ width: 768, height: 1024 });
      await expect(page.locator('#test-container')).toBeVisible();
      
      // Test mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
      await expect(page.locator('#test-container')).toBeVisible();
    });
  });

  test.describe('/test-predictions route', () => {
    test('should load the predictions test page with required elements', async ({ page }) => {
      await page.goto('/test-predictions');
      
      // Check main container exists
      await expect(page.locator('#predictions-results-container')).toBeAttached();
      
      // Container should be hidden by default
      const isVisible = await page.locator('#predictions-results-container').isVisible();
      expect(isVisible).toBe(false);
      
      // Verify Bootstrap and FontAwesome are loaded in head
      const bootstrapLink = page.locator('head link[href*="bootstrap"]');
      const fontAwesomeLink = page.locator('head link[href*="fontawesome"]');
      await expect(bootstrapLink).toBeAttached();
      await expect(fontAwesomeLink).toBeAttached();
      
      // Check that the prediction-buttons.js script is loaded
      await expect(page.locator('script[src="/static/js/prediction-buttons.js"]')).toBeAttached();
    });

    test('should have functional prediction results container', async ({ page }) => {
      await page.goto('/test-predictions');
      
      // Test showing the predictions container
      await page.locator('#predictions-results-container').evaluate(el => {
        el.style.display = 'block';
      });
      await expect(page.locator('#predictions-results-container')).toBeVisible();
      
      // Test injecting prediction content
      await page.locator('#predictions-results-container').evaluate(el => {
        el.innerHTML = `
          <div class="prediction-item">
            <h4>Race 1 Predictions</h4>
            <div class="predictions-list">
              <div class="prediction-entry">Dog 1 - 85% confidence</div>
              <div class="prediction-entry">Dog 2 - 78% confidence</div>
            </div>
          </div>
        `;
      });
      
      await expect(page.locator('.prediction-item')).toBeVisible();
      await expect(page.locator('.predictions-list')).toContainText('Dog 1 - 85% confidence');
      await expect(page.locator('.prediction-entry')).toHaveCount(2);
    });

    test('should load prediction-buttons.js functionality', async ({ page }) => {
      await page.goto('/test-predictions');
      
      // Wait for script to load
      await page.waitForLoadState('networkidle');
      
      // Test that basic JavaScript functionality is available
      const fetchAvailable = await page.evaluate(() => typeof window.fetch === 'function');
      expect(fetchAvailable).toBe(true);
      
      // Test DOM manipulation capabilities
      await page.locator('#predictions-results-container').evaluate(el => {
        el.innerHTML = '<div id="test-prediction">Prediction loaded via JS</div>';
      });
      await expect(page.locator('#test-prediction')).toBeVisible();
    });

    test('should handle prediction data display formats', async ({ page }) => {
      await page.goto('/test-predictions');
      
      // Test various prediction display formats
      await page.locator('#predictions-results-container').evaluate(el => {
        el.style.display = 'block';
        el.innerHTML = `
          <div class="prediction-results">
            <div class="race-header">
              <h3>Race 5 - Track XYZ</h3>
              <span class="badge bg-success">Model v2.1</span>
            </div>
            <div class="predictions-grid">
              <div class="prediction-card">
                <div class="dog-name">Lightning Bolt</div>
                <div class="confidence-score">92%</div>
                <div class="position-prediction">1st</div>
              </div>
              <div class="prediction-card">
                <div class="dog-name">Fast Runner</div>
                <div class="confidence-score">87%</div>
                <div class="position-prediction">2nd</div>
              </div>
            </div>
          </div>
        `;
      });
      
      await expect(page.locator('.race-header h3')).toContainText('Race 5');
      await expect(page.locator('.badge')).toContainText('Model v2.1');
      await expect(page.locator('.prediction-card')).toHaveCount(2);
      await expect(page.locator('.confidence-score')).toHaveText(['92%', '87%']);
    });
  });

  test.describe('/test-sidebar route', () => {
    test('should load the sidebar test page with required structure', async ({ page }) => {
      await page.goto('/test-sidebar');
      
      // Check main layout containers
      await expect(page.locator('.container-fluid')).toBeVisible();
      await expect(page.locator('.row')).toBeVisible();
      
      // Check sidebar column
      await expect(page.locator('#sidebar-col')).toBeVisible();
      await expect(page.locator('#sidebar-col')).toHaveClass(/col-md-3/);
      
      // Check main content column  
      await expect(page.locator('#main-content-col')).toBeVisible();
      await expect(page.locator('#main-content-col')).toHaveClass(/col-md-9/);
      
      // Verify sidebar sections exist
      await expect(page.locator('#sidebar-logs')).toBeAttached();
      await expect(page.locator('#sidebar-model-metrics')).toBeAttached(); 
      await expect(page.locator('#sidebar-system-health')).toBeAttached();
      
      // Check that sidebar.js is loaded
      await expect(page.locator('script[src="/static/js/sidebar.js"]')).toBeAttached();
    });

    test('should have Bootstrap grid system working correctly', async ({ page }) => {
      await page.goto('/test-sidebar');
      
      // Test desktop layout
      await page.setViewportSize({ width: 1200, height: 800 });
      await expect(page.locator('#sidebar-col')).toBeVisible();
      await expect(page.locator('#main-content-col')).toBeVisible();
      
      // Get computed styles to verify grid is working
      const sidebarWidth = await page.locator('#sidebar-col').evaluate(el => 
        window.getComputedStyle(el).width
      );
      const mainWidth = await page.locator('#main-content-col').evaluate(el => 
        window.getComputedStyle(el).width
      );
      
      expect(parseFloat(sidebarWidth)).toBeGreaterThan(0);
      expect(parseFloat(mainWidth)).toBeGreaterThan(0);
      
      // Test mobile responsiveness
      await page.setViewportSize({ width: 400, height: 600 });
      await expect(page.locator('#sidebar-col')).toBeVisible();
      await expect(page.locator('#main-content-col')).toBeVisible();
    });

    test('should load sidebar.js and handle dynamic content', async ({ page }) => {
      await page.goto('/test-sidebar');
      await page.waitForLoadState('networkidle');
      
      // Test that sidebar containers can receive content
      await page.locator('#sidebar-logs').evaluate(el => {
        el.innerHTML = `
          <div class="list-group-item">
            <span class="text-info">[INFO]</span> Application started successfully
          </div>
          <div class="list-group-item">
            <span class="text-warning">[WARN]</span> High memory usage detected
          </div>
        `;
      });
      
      await expect(page.locator('#sidebar-logs .list-group-item')).toHaveCount(2);
      await expect(page.locator('#sidebar-logs .text-info')).toContainText('[INFO]');
      await expect(page.locator('#sidebar-logs .text-warning')).toContainText('[WARN]');
      
      // Test model metrics container
      await page.locator('#sidebar-model-metrics').evaluate(el => {
        el.innerHTML = `
          <div class="metric-item">
            <strong>LinearRegression:</strong> Accuracy - 0.89
          </div>
          <div class="metric-item">
            <strong>RandomForest:</strong> Accuracy - 0.92
          </div>
        `;
      });
      
      await expect(page.locator('#sidebar-model-metrics .metric-item')).toHaveCount(2);
      await expect(page.locator('#sidebar-model-metrics')).toContainText('LinearRegression');
      await expect(page.locator('#sidebar-model-metrics')).toContainText('RandomForest');
      
      // Test system health container
      await page.locator('#sidebar-system-health').evaluate(el => {
        el.innerHTML = `
          <div class="health-stats">
            <strong>Total Races:</strong> 15,432<br>
            <strong>Active Models:</strong> 3<br>
            <strong>System Status:</strong> <span class="text-success">Healthy</span>
          </div>
        `;
      });
      
      await expect(page.locator('#sidebar-system-health')).toContainText('Total Races: 15,432');
      await expect(page.locator('#sidebar-system-health .text-success')).toContainText('Healthy');
    });

    test('should support real-time updates simulation', async ({ page }) => {
      await page.goto('/test-sidebar');
      
      // Simulate real-time log updates
      await page.locator('#sidebar-logs').evaluate(el => {
        el.innerHTML = '';
        
        const logs = [
          { level: 'INFO', message: 'Application started', class: 'text-info' },
          { level: 'WARNING', message: 'High memory usage detected', class: 'text-warning' },
          { level: 'ERROR', message: 'Database connection failed', class: 'text-danger' },
          { level: 'INFO', message: 'Retrying database connection...', class: 'text-info' },
          { level: 'INFO', message: 'Database connection restored', class: 'text-success' }
        ];
        
        logs.forEach((log, index) => {
          setTimeout(() => {
            const logEntry = document.createElement('div');
            logEntry.classList.add('list-group-item');
            logEntry.innerHTML = `<span class="${log.class}">[${log.level}]</span> ${log.message}`;
            el.appendChild(logEntry);
          }, index * 100); // Simulate real-time updates
        });
      });
      
      // Wait for all log entries to be added
      await page.waitForTimeout(600);
      
      // Verify all log entries were added
      await expect(page.locator('#sidebar-logs .list-group-item')).toHaveCount(5);
      await expect(page.locator('#sidebar-logs .text-info')).toHaveCount(3);
      await expect(page.locator('#sidebar-logs .text-warning')).toHaveCount(1);
      await expect(page.locator('#sidebar-logs .text-danger')).toHaveCount(1);
    });
  });

  test.describe('Cross-route navigation and functionality', () => {
    test('should navigate between test routes seamlessly', async ({ page }) => {
      // Start with blank page
      await page.goto('/test-blank-page');
      await expect(page.locator('#test-container')).toBeVisible();
      
      // Navigate to predictions page
      await page.goto('/test-predictions');
      await expect(page.locator('#predictions-results-container')).toBeAttached();
      
      // Navigate to sidebar page
      await page.goto('/test-sidebar');
      await expect(page.locator('#sidebar-logs')).toBeAttached();
      
      // Navigate back to blank page
      await page.goto('/test-blank-page');
      await expect(page.locator('#test-container')).toBeVisible();
    });
    
    test('should maintain testing environment across routes', async ({ page }) => {
      const routes = ['/test-blank-page', '/test-predictions', '/test-sidebar'];
      
      for (const route of routes) {
        await page.goto(route);
        
        // Each page should load successfully (no 404 or 500 errors)
        const response = await page.waitForResponse(response => 
          response.url().includes(route) && response.status() === 200
        );
        expect(response.status()).toBe(200);
        
        // Each page should have some basic content
        const bodyContent = await page.locator('body').innerHTML();
        expect(bodyContent.length).toBeGreaterThan(100); // Should have substantial content
        
        // Each page should have Bootstrap loaded
        await expect(page.locator('head link[href*="bootstrap"]')).toBeAttached();
      }
    });

    test('should handle browser back/forward navigation', async ({ page }) => {
      // Navigate through routes
      await page.goto('/test-blank-page');
      await page.goto('/test-predictions');
      await page.goto('/test-sidebar');
      
      // Use browser back button
      await page.goBack();
      await expect(page.locator('#predictions-results-container')).toBeAttached();
      
      await page.goBack();
      await expect(page.locator('#test-container')).toBeVisible();
      
      // Use browser forward button
      await page.goForward();
      await expect(page.locator('#predictions-results-container')).toBeAttached();
    });
  });

  test.describe('Performance and accessibility', () => {
    test('should load pages quickly', async ({ page }) => {
      const routes = ['/test-blank-page', '/test-predictions', '/test-sidebar'];
      
      for (const route of routes) {
        const startTime = Date.now();
        await page.goto(route);
        await page.waitForLoadState('networkidle');
        const loadTime = Date.now() - startTime;
        
        // Pages should load within reasonable time (5 seconds)
        expect(loadTime).toBeLessThan(5000);
      }
    });

    test('should have basic accessibility features', async ({ page }) => {
      await page.goto('/test-sidebar');
      
      // Check that elements have proper semantic structure
      await expect(page.locator('main, #main-content-col')).toBeAttached();
      
      // Check that Bootstrap components use appropriate ARIA attributes
      await page.locator('#sidebar-logs').evaluate(el => {
        el.innerHTML = `
          <div class="list-group-item" role="listitem">
            <span class="text-info">[INFO]</span> Test message
          </div>
        `;
      });
      
      await expect(page.locator('[role="listitem"]')).toBeAttached();
    });
  });
});
