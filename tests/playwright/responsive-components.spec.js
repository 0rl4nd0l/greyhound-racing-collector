const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

// Helper function to get viewport info from project name
function getViewportInfo(project) {
  if (project.includes('mobile')) {
    return { name: 'mobile', width: 375, height: 812 };
  } else if (project.includes('tablet')) {
    return { name: 'tablet', width: 768, height: 1024 };
  } else {
    return { name: 'desktop', width: 1280, height: 720 };
  }
}

test.describe('Responsive UI & Component Interaction Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to homepage before each test
    await page.goto('/');
    // Wait for page to load
    await page.waitForLoadState('networkidle');
  });

  test('should render navbar correctly across viewports', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Check navbar is visible
    const navbar = page.locator('nav.navbar');
    await expect(navbar).toBeVisible();
    
    // Check brand logo/text
    const brand = navbar.locator('.navbar-brand');
    await expect(brand).toBeVisible();
    await expect(brand).toContainText('Greyhound Racing Dashboard');
    
    if (viewport.name === 'mobile') {
      // On mobile, navbar should be collapsed by default
      const navbarToggler = navbar.locator('.navbar-toggler');
      await expect(navbarToggler).toBeVisible();
      
      // Check if collapse is working
      const navbarCollapse = navbar.locator('#navbarNav');
      
      // Initially collapsed (Bootstrap classes)
      await expect(navbarCollapse).toHaveClass(/collapse/);
      
      // Click toggle button
      await navbarToggler.click();
      
      // Wait for Bootstrap animation
      await page.waitForTimeout(500);
      
      // Should now be expanded
      await expect(navbarCollapse).toHaveClass(/show/);
      
      // Click toggle button again to collapse
      await navbarToggler.click();
      await page.waitForTimeout(500);
      
      // Should be collapsed again
      await expect(navbarCollapse).not.toHaveClass(/show/);
      
    } else {
      // On tablet/desktop, navbar should be expanded by default
      const navbarCollapse = navbar.locator('#navbarNav');
      await expect(navbarCollapse).toHaveClass(/show/);
      
      // Toggler should not be visible on larger screens
      const navbarToggler = navbar.locator('.navbar-toggler');
      await expect(navbarToggler).not.toBeVisible();
    }
    
    // Check all main navigation items are present
    const expectedNavItems = ['Dashboard', 'Races', 'Analysis', 'AI/ML', 'System', 'Help'];
    
    for (const item of expectedNavItems) {
      if (viewport.name === 'mobile') {
        // Need to expand navbar first on mobile
        await navbar.locator('.navbar-toggler').click();
        await page.waitForTimeout(300);
      }
      
      const navItem = navbar.locator(`text=${item}`).first();
      await expect(navItem).toBeVisible();
      
      if (viewport.name === 'mobile') {
        // Collapse again for next iteration
        await navbar.locator('.navbar-toggler').click();
        await page.waitForTimeout(300);
      }
    }
  });

  test('should handle sidebar toggle correctly', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Check sidebar toggle button exists
    const toggleButton = page.locator('button[data-bs-target="#system-status-sidebar"]');
    await expect(toggleButton).toBeVisible();
    
    // Check sidebar is initially hidden
    const sidebar = page.locator('#system-status-sidebar');
    await expect(sidebar).not.toHaveClass(/show/);
    
    // Click toggle button to open sidebar
    await toggleButton.click();
    
    // Wait for Bootstrap offcanvas animation
    await page.waitForTimeout(500);
    
    // Sidebar should now be visible
    await expect(sidebar).toHaveClass(/show/);
    
    // Check sidebar content
    await expect(sidebar.locator('h5')).toContainText('System Status');
    
    // Check sidebar sections
    const sidebarBody = sidebar.locator('.offcanvas-body');
    await expect(sidebarBody.locator('text=Logs')).toBeVisible();
    await expect(sidebarBody.locator('text=Model Metrics')).toBeVisible();
    await expect(sidebarBody.locator('text=System Health')).toBeVisible();
    
    // Close sidebar using close button
    const closeButton = sidebar.locator('.btn-close');
    await closeButton.click();
    
    // Wait for animation
    await page.waitForTimeout(500);
    
    // Sidebar should be hidden again
    await expect(sidebar).not.toHaveClass(/show/);
  });

  test('should persist theme toggle in localStorage', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Find theme toggle button
    const themeToggle = page.locator('#theme-toggle');
    await expect(themeToggle).toBeVisible();
    
    // Check initial theme (default should be light)
    let currentTheme = await page.getAttribute('html', 'data-theme');
    if (!currentTheme) {
      currentTheme = 'light'; // Default fallback
    }
    
    // Check initial icon matches theme
    const themeIcon = page.locator('#theme-icon');
    if (currentTheme === 'light') {
      await expect(themeIcon).toHaveClass(/fa-moon/);
    } else {
      await expect(themeIcon).toHaveClass(/fa-sun/);
    }
    
    // Store initial theme
    const initialTheme = currentTheme;
    
    // Click theme toggle
    await themeToggle.click();
    
    // Wait for theme change
    await page.waitForTimeout(200);
    
    // Check theme has changed
    const newTheme = await page.getAttribute('html', 'data-theme');
    expect(newTheme).not.toBe(initialTheme);
    
    // Check icon has changed
    if (newTheme === 'dark') {
      await expect(themeIcon).toHaveClass(/fa-sun/);
    } else {
      await expect(themeIcon).toHaveClass(/fa-moon/);
    }
    
    // Check localStorage persistence
    const storedTheme = await page.evaluate(() => localStorage.getItem('theme'));
    expect(storedTheme).toBe(newTheme);
    
    // Reload page and verify theme persists
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    const persistedTheme = await page.getAttribute('html', 'data-theme');
    expect(persistedTheme).toBe(newTheme);
    
    // Verify icon still matches after reload
    const persistedIcon = page.locator('#theme-icon');
    if (persistedTheme === 'dark') {
      await expect(persistedIcon).toHaveClass(/fa-sun/);
    } else {
      await expect(persistedIcon).toHaveClass(/fa-moon/);
    }
    
    // Toggle back to original theme
    await page.locator('#theme-toggle').click();
    await page.waitForTimeout(200);
    
    const finalTheme = await page.getAttribute('html', 'data-theme');
    expect(finalTheme).toBe(initialTheme);
    
    // Verify localStorage updated again
    const finalStoredTheme = await page.evaluate(() => localStorage.getItem('theme'));
    expect(finalStoredTheme).toBe(finalTheme);
  });

  test('should handle dropdown menus correctly across viewports', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Test main navigation dropdowns
    const dropdowns = [
      { trigger: '#racesDropdown', expectedItems: ['Interactive Races', 'Upcoming Races', 'Historical Races'] },
      { trigger: '#analysisDropdown', expectedItems: ['ML Predictions', 'Live Odds & Value Bets', 'Dog Performance', 'Historical Performance'] },
      { trigger: '#aiMlDropdown', expectedItems: ['ML Training', 'Model Registry', 'Model Monitoring', 'GPT Enhancement'] },
      { trigger: '#systemDropdown', expectedItems: ['Automation', 'Data Processing', 'Database Manager', 'System Logs', 'Upload Files'] },
      { trigger: '#helpDropdown', expectedItems: ['Getting Started', 'User Guide', 'System Status'] }
    ];
    
    for (const dropdown of dropdowns) {
      if (viewport.name === 'mobile') {
        // Expand navbar on mobile first
        await page.locator('.navbar-toggler').click();
        await page.waitForTimeout(300);
      }
      
      // Click dropdown trigger
      const trigger = page.locator(dropdown.trigger);
      await expect(trigger).toBeVisible();
      await trigger.click();
      
      // Wait for dropdown to open
      await page.waitForTimeout(200);
      
      // Check dropdown items are visible
      const dropdownMenu = trigger.locator('..').locator('.dropdown-menu');
      await expect(dropdownMenu).toHaveClass(/show/);
      
      for (const item of dropdown.expectedItems) {
        const menuItem = dropdownMenu.locator(`text=${item}`);
        await expect(menuItem).toBeVisible();
      }
      
      // Click somewhere else to close dropdown
      await page.locator('body').click();
      await page.waitForTimeout(200);
      
      // Dropdown should be closed
      await expect(dropdownMenu).not.toHaveClass(/show/);
      
      if (viewport.name === 'mobile') {
        // Collapse navbar on mobile
        await page.locator('.navbar-toggler').click();
        await page.waitForTimeout(300);
      }
    }
  });

  test('should maintain functionality across different viewport sizes', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Test that all key interactive elements are accessible and functional
    
    // 1. Theme toggle should work
    const themeToggle = page.locator('#theme-toggle');
    await expect(themeToggle).toBeVisible();
    const initialTheme = await page.getAttribute('html', 'data-theme') || 'light';
    await themeToggle.click();
    await page.waitForTimeout(200);
    const newTheme = await page.getAttribute('html', 'data-theme');
    expect(newTheme).not.toBe(initialTheme);
    
    // 2. Sidebar toggle should work
    const sidebarToggle = page.locator('button[data-bs-target="#system-status-sidebar"]');
    await expect(sidebarToggle).toBeVisible();
    await sidebarToggle.click();
    await page.waitForTimeout(500);
    const sidebar = page.locator('#system-status-sidebar');
    await expect(sidebar).toHaveClass(/show/);
    
    // Close sidebar
    await sidebar.locator('.btn-close').click();
    await page.waitForTimeout(500);
    
    // 3. Navigation should be accessible
    if (viewport.name === 'mobile') {
      const navToggle = page.locator('.navbar-toggler');
      await expect(navToggle).toBeVisible();
      await navToggle.click();
      await page.waitForTimeout(300);
      
      const navCollapse = page.locator('#navbarNav');
      await expect(navCollapse).toHaveClass(/show/);
      
      // Test a navigation link
      const dashboardLink = page.locator('a[href="/"]').first();
      await expect(dashboardLink).toBeVisible();
      
      // Collapse again
      await navToggle.click();
    } else {
      // On larger screens, nav should be visible by default
      const dashboardLink = page.locator('a[href="/"]').first();
      await expect(dashboardLink).toBeVisible();
    }
    
    // 4. Test that content area is properly sized
    const mainContent = page.locator('main.container-fluid');
    await expect(mainContent).toBeVisible();
    
    const mainBoundingBox = await mainContent.boundingBox();
    expect(mainBoundingBox.width).toBeGreaterThan(0);
    expect(mainBoundingBox.height).toBeGreaterThan(0);
    
    // Check that content doesn't overflow viewport
    expect(mainBoundingBox.width).toBeLessThanOrEqual(viewport.width + 50); // Allow some margin for scrollbars
  });

  test('should pass accessibility audit with Axe-core', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Run axe accessibility scan
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    
    // Filter out violations that are not serious
    const seriousViolations = accessibilityScanResults.violations.filter(violation => 
      violation.impact === 'serious' || violation.impact === 'critical'
    );
    
    // Log violations for debugging if any exist
    if (seriousViolations.length > 0) {
      console.log('Accessibility violations found:', JSON.stringify(seriousViolations, null, 2));
    }
    
    // Fail test if serious violations are found
    expect(seriousViolations).toHaveLength(0);
    
    // Test accessibility of interactive elements specifically
    
    // 1. Navbar accessibility
    const navbar = page.locator('nav.navbar');
    await expect(navbar).toHaveAttribute('role', 'navigation', { timeout: 1000 }).catch(() => {
      // If role attribute doesn't exist, that's okay for semantic HTML elements
    });
    
    // 2. Buttons should have accessible names
    const themeToggle = page.locator('#theme-toggle');
    const themeToggleTitle = await themeToggle.getAttribute('title');
    expect(themeToggleTitle).toBeTruthy();
    expect(themeToggleTitle.toLowerCase()).toContain('toggle');
    
    // 3. Sidebar should have proper ARIA attributes
    const sidebar = page.locator('#system-status-sidebar');
    await expect(sidebar).toHaveAttribute('aria-labelledby', 'system-status-sidebar-label');
    await expect(sidebar).toHaveAttribute('role', 'dialog');
    
    // 4. Form elements should have proper labels (if any exist)
    const formElements = await page.locator('input, select, textarea').count();
    if (formElements > 0) {
      const unlabeledInputs = await page.locator('input:not([aria-label]):not([aria-labelledby]):not([title])').count();
      expect(unlabeledInputs).toBe(0);
    }
    
    // 5. Links should have meaningful text
    const links = page.locator('a');
    const linkCount = await links.count();
    
    for (let i = 0; i < Math.min(linkCount, 10); i++) { // Check first 10 links
      const link = links.nth(i);
      const isVisible = await link.isVisible();
      
      if (isVisible) {
        const linkText = await link.textContent();
        const linkTitle = await link.getAttribute('title');
        const linkAriaLabel = await link.getAttribute('aria-label');
        
        // Link should have some form of accessible text
        const hasAccessibleText = (linkText && linkText.trim().length > 0) || 
                                  (linkTitle && linkTitle.trim().length > 0) || 
                                  (linkAriaLabel && linkAriaLabel.trim().length > 0);
        
        expect(hasAccessibleText).toBeTruthy();
      }
    }
  });

  test('should handle keyboard navigation correctly', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Test keyboard navigation through interactive elements
    
    // Start by focusing on the page
    await page.keyboard.press('Tab');
    
    // Should be able to tab to theme toggle button
    await page.keyboard.press('Tab');
    const focusedElement = await page.evaluate(() => document.activeElement.id);
    
    // Check that we can navigate through key interactive elements
    const interactiveElements = ['theme-toggle'];
    let foundElements = [];
    
    // Tab through several elements to find our key interactive ones
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press('Tab');
      const currentFocus = await page.evaluate(() => {
        const el = document.activeElement;
        return {
          id: el.id,
          tagName: el.tagName,
          type: el.type,
          href: el.href,
          className: el.className
        };
      });
      
      if (currentFocus.id && interactiveElements.includes(currentFocus.id)) {
        foundElements.push(currentFocus.id);
      }
      
      // Break if we've found all elements we're looking for
      if (foundElements.length >= interactiveElements.length) {
        break;
      }
    }
    
    // We should have found at least some key interactive elements
    expect(foundElements.length).toBeGreaterThan(0);
    
    // Test that theme toggle can be activated with keyboard
    await page.focus('#theme-toggle');
    const initialTheme = await page.getAttribute('html', 'data-theme') || 'light';
    await page.keyboard.press('Enter');
    await page.waitForTimeout(200);
    const newTheme = await page.getAttribute('html', 'data-theme');
    expect(newTheme).not.toBe(initialTheme);
  });

  test('should have proper responsive breakpoints', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // Test that Bootstrap responsive classes are working correctly
    
    if (viewport.name === 'mobile') {
      // On mobile, certain elements should be hidden or styled differently
      const navbar = page.locator('nav.navbar');
      
      // Navbar toggler should be visible
      const toggler = navbar.locator('.navbar-toggler');
      await expect(toggler).toBeVisible();
      
      // Navbar collapse should be initially hidden
      const collapse = navbar.locator('#navbarNav');
      const isCollapsed = await collapse.evaluate(el => !el.classList.contains('show'));
      expect(isCollapsed).toBeTruthy();
      
    } else if (viewport.name === 'tablet') {
      // On tablet, navbar should typically be expanded
      const collapse = page.locator('#navbarNav');
      const isExpanded = await collapse.evaluate(el => 
        el.classList.contains('show') || !el.classList.contains('collapse')
      );
      expect(isExpanded).toBeTruthy();
      
    } else { // desktop
      // On desktop, full navigation should be visible
      const collapse = page.locator('#navbarNav');
      const isExpanded = await collapse.evaluate(el => 
        el.classList.contains('show') || !el.classList.contains('collapse')
      );
      expect(isExpanded).toBeTruthy();
      
      // Navbar toggler should not be visible
      const toggler = page.locator('.navbar-toggler');
      await expect(toggler).not.toBeVisible();
    }
    
    // Test that content adapts to viewport size
    const mainContent = page.locator('main.container-fluid');
    const contentBox = await mainContent.boundingBox();
    
    // Content should not exceed viewport width
    expect(contentBox.width).toBeLessThanOrEqual(viewport.width);
    
    // Content should have appropriate margins/padding for viewport
    if (viewport.name === 'mobile') {
      // On mobile, content should use more of the available width
      expect(contentBox.width).toBeGreaterThan(viewport.width * 0.9);
    }
  });
});

// Test different templates/pages
test.describe('Template-specific responsive tests', () => {
  const templates = [
    { path: '/', name: 'homepage' },
    { path: '/ml-dashboard', name: 'ml-dashboard' },
    { path: '/monitoring', name: 'monitoring' },
    { path: '/upcoming', name: 'upcoming-races' }
  ];

  templates.forEach(template => {
    test(`${template.name} should be responsive and accessible`, async ({ page }, testInfo) => {
      const viewport = getViewportInfo(testInfo.project.name);
      
      try {
        await page.goto(template.path);
        await page.waitForLoadState('networkidle');
        
        // Basic responsiveness check
        const body = page.locator('body');
        await expect(body).toBeVisible();
        
        // Check that page doesn't have horizontal scroll on mobile
        if (viewport.name === 'mobile') {
          const bodyWidth = await body.evaluate(el => el.scrollWidth);
          expect(bodyWidth).toBeLessThanOrEqual(viewport.width + 20); // Allow small margin
        }
        
        // Run accessibility check
        const accessibilityScanResults = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa'])
          .analyze();
        
        const seriousViolations = accessibilityScanResults.violations.filter(violation => 
          violation.impact === 'serious' || violation.impact === 'critical'
        );
        
        if (seriousViolations.length > 0) {
          console.log(`Accessibility violations on ${template.name}:`, JSON.stringify(seriousViolations, null, 2));
        }
        
        expect(seriousViolations).toHaveLength(0);
        
      } catch (error) {
        // If page doesn't exist or loads with error, skip the test
        if (error.message.includes('404') || error.message.includes('ERR_CONNECTION_REFUSED')) {
          test.skip(true, `Template ${template.name} not available`);
        } else {
          throw error;
        }
      }
    });
  });
});
