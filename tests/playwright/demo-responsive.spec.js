const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;
const fs = require('fs');
const path = require('path');

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

// Create a demo HTML file for testing
const demoHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Greyhound Racing Dashboard - Demo</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    
    <style>
        [data-theme="dark"] {
            background-color: #121212;
            color: #ffffff;
        }
        
        [data-theme="dark"] .navbar {
            background-color: #1f1f1f !important;
        }
        
        [data-theme="dark"] .card {
            background-color: #1e1e1e;
            border-color: #333;
        }
        
        .alert-fixed {
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 1050;
            max-width: 400px;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">
                <i class="fas fa-dog" aria-hidden="true"></i>
                Greyhound Racing Dashboard
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link active" href="/">
                            <i class="fas fa-home" aria-hidden="true"></i> Dashboard
                        </a>
                    </li>
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="racesDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="fas fa-flag-checkered" aria-hidden="true"></i> Races
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="/interactive-races">Interactive Races</a></li>
                            <li><a class="dropdown-item" href="/upcoming">Upcoming Races</a></li>
                            <li><a class="dropdown-item" href="/races">Historical Races</a></li>
                        </ul>
                    </li>
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="analysisDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="fas fa-chart-bar" aria-hidden="true"></i> Analysis
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="/ml-dashboard">ML Predictions</a></li>
                            <li><a class="dropdown-item" href="/odds_dashboard">Live Odds & Value Bets</a></li>
                            <li><a class="dropdown-item" href="/dogs">Dog Performance</a></li>
                            <li><a class="dropdown-item" href="/enhanced_analysis">Historical Performance</a></li>
                        </ul>
                    </li>
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="aiMlDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="fas fa-brain" aria-hidden="true"></i> AI/ML
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="/ml-training">ML Training</a></li>
                            <li><a class="dropdown-item" href="/model_registry">Model Registry</a></li>
                            <li><a class="dropdown-item" href="/monitoring">Model Monitoring</a></li>
                            <li><a class="dropdown-item" href="/gpt-enhancement">GPT Enhancement</a></li>
                        </ul>
                    </li>
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="systemDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="fas fa-cogs" aria-hidden="true"></i> System
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="/automation">Automation</a></li>
                            <li><a class="dropdown-item" href="/scraping">Data Processing</a></li>
                            <li><a class="dropdown-item" href="/database-manager">Database Manager</a></li>
                            <li><a class="dropdown-item" href="/logs">System Logs</a></li>
                            <li><a class="dropdown-item" href="/upload">Upload Files</a></li>
                        </ul>
                    </li>
                    <li class="nav-item">
                        <button class="btn btn-outline-light btn-sm me-2" onclick="toggleTheme()" id="theme-toggle" title="Toggle dark mode" aria-label="Toggle dark/light theme">
                            <i class="fas fa-moon" id="theme-icon" aria-hidden="true"></i>
                            <span class="visually-hidden">Toggle dark/light theme</span>
                        </button>
                    </li>
                </ul>
                <ul class="navbar-nav">
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="helpDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="fas fa-question-circle" aria-hidden="true"></i> Help
                        </a>
                        <ul class="dropdown-menu dropdown-menu-end">
                            <li><a class="dropdown-item" href="/upcoming">
                                <i class="fas fa-play" aria-hidden="true"></i> Getting Started
                            </a></li>
                            <li><a class="dropdown-item" href="/scraping">
                                <i class="fas fa-book" aria-hidden="true"></i> User Guide
                            </a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="/logs">
                                <i class="fas fa-bug" aria-hidden="true"></i> System Status
                            </a></li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    
    <!-- Alert Container -->
    <div id="alertContainer"></div>
    
    <!-- Main Content -->
    <main class="container-fluid" role="main">
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h1 class="card-title">Dashboard</h1>
                    </div>
                    <div class="card-body">
                        <p>Welcome to the Greyhound Racing Dashboard - Test Version</p>
                        <div class="row">
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Recent Races</h5>
                                        <p class="card-text">View recent race results</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Predictions</h5>
                                        <p class="card-text">ML-powered predictions</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Analytics</h5>
                                        <p class="card-text">Performance analytics</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Sidebar -->
    <aside class="offcanvas offcanvas-end" tabindex="-1" id="system-status-sidebar" aria-labelledby="system-status-sidebar-label" aria-modal="true" role="dialog">
        <div class="offcanvas-header">
            <h5 id="system-status-sidebar-label">System Status</h5>
            <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas" aria-label="Close"></button>
        </div>
        <div class="offcanvas-body">
            <!-- Logs Section -->
            <h6><i class="fas fa-file-alt" aria-hidden="true"></i> Logs</h6>
            <div id="sidebar-logs" class="list-group" style="max-height: 300px; overflow-y: auto;">
                <div class="list-group-item">System started</div>
                <div class="list-group-item">Model loaded</div>
            </div>
            <hr>
            <!-- Model Metrics Section -->
            <h6><i class="fas fa-robot" aria-hidden="true"></i> Model Metrics</h6>
            <div id="sidebar-model-metrics">
                <p>Accuracy: 85%</p>
                <p>Last updated: Now</p>
            </div>
            <hr>
            <!-- System Health Section -->
            <h6><i class="fas fa-heartbeat" aria-hidden="true"></i> System Health</h6>
            <div id="sidebar-system-health">
                <span class="badge bg-success">All systems operational</span>
            </div>
        </div>
    </aside>

    <!-- Toggle Button -->
    <button class="btn btn-primary position-fixed bottom-0 end-0 m-3" type="button" data-bs-toggle="offcanvas" data-bs-target="#system-status-sidebar" aria-controls="system-status-sidebar">
        <i class="fas fa-tachometer-alt" aria-hidden="true"></i> Status
    </button>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Global JavaScript -->
    <script>
        // Theme Management
        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update the theme toggle icon
            const themeIcon = document.getElementById('theme-icon');
            if (themeIcon) {
                themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
            
            // Update the toggle button title
            const themeToggle = document.getElementById('theme-toggle');
            if (themeToggle) {
                themeToggle.title = newTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
            }
        }

        // Load saved theme on page load
        function loadTheme() {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            // Update the theme toggle icon
            const themeIcon = document.getElementById('theme-icon');
            if (themeIcon) {
                themeIcon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
            
            // Update the toggle button title
            const themeToggle = document.getElementById('theme-toggle');
            if (themeToggle) {
                themeToggle.title = savedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
            }
        }

        // Load theme immediately to prevent flash
        loadTheme();

        // Global utility functions
        function showAlert(message, type = 'info', duration = 5000) {
            const alertContainer = document.getElementById('alertContainer');
            const alert = document.createElement('div');
            alert.className = \`alert alert-\${type} alert-dismissible fade show alert-fixed\`;
            alert.innerHTML = \`
                \${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            \`;
            alertContainer.appendChild(alert);

            // Auto-remove after duration
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, duration);
        }

        // Initialize Bootstrap tooltips
        document.addEventListener('DOMContentLoaded', function() {
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"], [title]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                if (!tooltipTriggerEl.hasAttribute('data-bs-toggle') && tooltipTriggerEl.hasAttribute('title')) {
                    tooltipTriggerEl.setAttribute('data-bs-toggle', 'tooltip');
                }
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        });
    </script>
</body>
</html>`;

test.describe('Demo Responsive UI & Component Tests', () => {
  let tempFile;

  test.beforeAll(async () => {
    // Create temporary HTML file
    tempFile = path.join(__dirname, 'demo-page.html');
    fs.writeFileSync(tempFile, demoHTML);
  });

  test.afterAll(async () => {
    // Clean up temporary file
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  });

  test.beforeEach(async ({ page }) => {
    // Navigate to the demo HTML file
    await page.goto(`file://${tempFile}`);
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
    
    // On mobile, expand navbar first to access theme toggle
    if (viewport.name === 'mobile') {
      const navbarToggler = page.locator('.navbar-toggler');
      await navbarToggler.click();
      await page.waitForTimeout(500);
    }
    
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
  });

  test('should pass accessibility audit with Axe-core', async ({ page }, testInfo) => {
    const viewport = getViewportInfo(testInfo.project.name);
    
    // On mobile, expand navbar first to access theme toggle
    if (viewport.name === 'mobile') {
      const navbarToggler = page.locator('.navbar-toggler');
      await navbarToggler.click();
      await page.waitForTimeout(500);
    }
    
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
    
    // 1. Buttons should have accessible names
    const themeToggle = page.locator('#theme-toggle');
    const themeToggleAriaLabel = await themeToggle.getAttribute('aria-label');
    expect(themeToggleAriaLabel).toBeTruthy();
    expect(themeToggleAriaLabel.toLowerCase()).toContain('toggle');
    
    // 2. Sidebar should have proper ARIA attributes
    const sidebar = page.locator('#system-status-sidebar');
    await expect(sidebar).toHaveAttribute('aria-labelledby', 'system-status-sidebar-label');
    await expect(sidebar).toHaveAttribute('role', 'dialog');
    
    // 3. Links should have meaningful text
    const links = page.locator('a');
    const linkCount = await links.count();
    
    for (let i = 0; i < Math.min(linkCount, 5); i++) { // Check first 5 links
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
