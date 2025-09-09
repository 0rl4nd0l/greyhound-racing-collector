/**
 * Database Manager Tabs
 * Accessible tabbed interface with keyboard navigation and ARIA support
 */

class DatabaseManagerTabs {
  constructor(containerSelector = '.db-tabs') {
    this.container = document.querySelector(containerSelector);
    if (!this.container) return;

    this.tabList = this.container.querySelector('[role="tablist"]');
    this.tabs = this.container.querySelectorAll('[role="tab"]');
    this.panels = this.container.querySelectorAll('[role="tabpanel"]');
    
    this.currentTabIndex = 0;
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.loadSavedTab();
    this.setupResizeObserver();
  }

  setupEventListeners() {
    // Tab click handlers
    this.tabs.forEach((tab, index) => {
      tab.addEventListener('click', (e) => {
        e.preventDefault();
        this.activateTab(index);
      });

      // Keyboard navigation
      tab.addEventListener('keydown', (e) => {
        this.handleKeyDown(e, index);
      });
    });

    // Hash change for URL routing
    window.addEventListener('hashchange', () => {
      this.handleHashChange();
    });

    // Save tab state on beforeunload
    window.addEventListener('beforeunload', () => {
      this.saveTabState();
    });
  }

  handleKeyDown(event, tabIndex) {
    let newIndex = tabIndex;

    switch (event.key) {
      case 'ArrowLeft':
      case 'ArrowUp':
        event.preventDefault();
        newIndex = tabIndex > 0 ? tabIndex - 1 : this.tabs.length - 1;
        this.focusTab(newIndex);
        break;

      case 'ArrowRight':
      case 'ArrowDown':
        event.preventDefault();
        newIndex = tabIndex < this.tabs.length - 1 ? tabIndex + 1 : 0;
        this.focusTab(newIndex);
        break;

      case 'Home':
        event.preventDefault();
        this.focusTab(0);
        break;

      case 'End':
        event.preventDefault();
        this.focusTab(this.tabs.length - 1);
        break;

      case 'Enter':
      case ' ':
        event.preventDefault();
        this.activateTab(tabIndex);
        break;
    }
  }

  focusTab(index) {
    this.tabs.forEach((tab, i) => {
      tab.tabIndex = i === index ? 0 : -1;
    });
    this.tabs[index].focus();
  }

  activateTab(index) {
    if (index < 0 || index >= this.tabs.length) return;

    // Update ARIA attributes
    this.tabs.forEach((tab, i) => {
      const isActive = i === index;
      tab.setAttribute('aria-selected', isActive);
      tab.tabIndex = isActive ? 0 : -1;
    });

    // Show/hide panels
    this.panels.forEach((panel, i) => {
      if (i === index) {
        panel.hidden = false;
        panel.setAttribute('aria-expanded', 'true');
        this.loadTabContent(panel, this.tabs[index]);
      } else {
        panel.hidden = true;
        panel.setAttribute('aria-expanded', 'false');
      }
    });

    // Update URL hash
    const tabId = this.tabs[index].getAttribute('aria-controls');
    if (tabId) {
      history.replaceState(null, '', `#${tabId}`);
    }

    this.currentTabIndex = index;
    this.announceTabChange(this.tabs[index]);
    
    // Trigger custom event
    this.container.dispatchEvent(new CustomEvent('tabchange', {
      detail: { 
        tabIndex: index, 
        tabElement: this.tabs[index],
        panelElement: this.panels[index]
      }
    }));
  }

  announceTabChange(tab) {
    // Create announcement for screen readers
    const announcement = `Tab ${tab.textContent.trim()} selected`;
    this.announce(announcement);
  }

  announce(message) {
    // Create a live region announcement
    let announcer = document.getElementById('db-tab-announcer');
    if (!announcer) {
      announcer = document.createElement('div');
      announcer.id = 'db-tab-announcer';
      announcer.setAttribute('aria-live', 'polite');
      announcer.setAttribute('aria-atomic', 'true');
      announcer.className = 'sr-only';
      announcer.style.position = 'absolute';
      announcer.style.left = '-10000px';
      announcer.style.width = '1px';
      announcer.style.height = '1px';
      announcer.style.overflow = 'hidden';
      document.body.appendChild(announcer);
    }

    announcer.textContent = message;
    
    // Clear after announcement
    setTimeout(() => {
      announcer.textContent = '';
    }, 1000);
  }

  loadTabContent(panel, tab) {
    const tabId = tab.getAttribute('aria-controls');
    
    // Skip if content already loaded
    if (panel.dataset.loaded === 'true') return;

    // Show loading state
    this.showTabLoading(panel);

    // Load content based on tab type
    switch (tabId) {
      case 'overview-panel':
        this.loadOverviewContent(panel);
        break;
      case 'realtime-panel':
        this.loadRealtimeContent(panel);
        break;
      case 'tables-panel':
        this.loadTablesContent(panel);
        break;
      case 'queries-panel':
        this.loadQueriesContent(panel);
        break;
      case 'migrations-panel':
        this.loadMigrationsContent(panel);
        break;
      case 'logs-panel':
        this.loadLogsContent(panel);
        break;
      case 'operations-panel':
        this.loadOperationsContent(panel);
        break;
      default:
        panel.dataset.loaded = 'true';
        break;
    }
  }

  showTabLoading(panel) {
    panel.innerHTML = `
      <div class=\"db-loading-container\" role=\"status\" aria-label=\"Loading content\">
        <div class=\"db-skeleton db-skeleton-card\"></div>
        <div style=\"margin-top: var(--space-6);\">
          <div class=\"db-skeleton db-skeleton-text\" style=\"width: 60%;\"></div>
          <div class=\"db-skeleton db-skeleton-text\" style=\"width: 80%;\"></div>
          <div class=\"db-skeleton db-skeleton-text\" style=\"width: 45%;\"></div>
        </div>
      </div>
    `;
  }

  async loadOverviewContent(panel) {
    try {
      const response = await fetch('/api/database/overview');
      const data = await response.json();
      
      if (data.success) {
        panel.innerHTML = this.renderOverviewContent(data);
      } else {
        panel.innerHTML = this.renderErrorContent('Failed to load overview data');
      }
    } catch (error) {
      console.error('Error loading overview:', error);
      panel.innerHTML = this.renderErrorContent('Failed to load overview data');
    } finally {
      panel.dataset.loaded = 'true';
    }
  }

  async loadRealtimeContent(panel) {
    try {
      // Initialize real-time connection
      if (window.DatabaseRealtime) {
        panel.innerHTML = this.renderRealtimeContent();
        new window.DatabaseRealtime(panel.querySelector('.realtime-container'));
      } else {
        panel.innerHTML = this.renderErrorContent('Real-time monitoring not available');
      }
    } catch (error) {
      console.error('Error loading realtime:', error);
      panel.innerHTML = this.renderErrorContent('Failed to initialize real-time monitoring');
    } finally {
      panel.dataset.loaded = 'true';
    }
  }

  async loadTablesContent(panel) {
    try {
      const response = await fetch('/api/database/tables');
      const data = await response.json();
      
      if (data.success) {
        panel.innerHTML = this.renderTablesContent(data);
        this.initializeTableInteractions(panel);
      } else {
        panel.innerHTML = this.renderErrorContent('Failed to load table data');
      }
    } catch (error) {
      console.error('Error loading tables:', error);
      panel.innerHTML = this.renderErrorContent('Failed to load table data');
    } finally {
      panel.dataset.loaded = 'true';
    }
  }

  async loadQueriesContent(panel) {
    try {
      const response = await fetch('/api/database/queries/slow?limit=50');
      const data = await response.json();
      
      if (data.success) {
        panel.innerHTML = this.renderQueriesContent(data);
        this.initializeQueryInteractions(panel);
      } else {
        panel.innerHTML = this.renderErrorContent('Failed to load query data');
      }
    } catch (error) {
      console.error('Error loading queries:', error);
      panel.innerHTML = this.renderErrorContent('Failed to load query data');
    } finally {
      panel.dataset.loaded = 'true';
    }
  }

  async loadMigrationsContent(panel) {
    try {
      const [migrationResponse, schemaResponse] = await Promise.all([
        fetch('/api/database/migrations/status'),
        fetch('/api/database/schema/drift')
      ]);
      
      const migrationData = await migrationResponse.json();
      const schemaData = await schemaResponse.json();
      
      panel.innerHTML = this.renderMigrationsContent(migrationData, schemaData);
    } catch (error) {
      console.error('Error loading migrations:', error);
      panel.innerHTML = this.renderErrorContent('Failed to load migration data');
    } finally {
      panel.dataset.loaded = 'true';
    }
  }

  async loadLogsContent(panel) {
    try {
      const response = await fetch('/api/database/logs?limit=100');
      const data = await response.json();
      
      if (data.success) {
        panel.innerHTML = this.renderLogsContent(data);
        this.initializeLogInteractions(panel);
      } else {
        panel.innerHTML = this.renderErrorContent('Failed to load log data');
      }
    } catch (error) {
      console.error('Error loading logs:', error);
      panel.innerHTML = this.renderErrorContent('Failed to load log data');
    } finally {
      panel.dataset.loaded = 'true';
    }
  }

  async loadOperationsContent(panel) {
    panel.innerHTML = this.renderOperationsContent();
    this.initializeOperationInteractions(panel);
    panel.dataset.loaded = 'true';
  }

  renderOverviewContent(data) {
    return `
      <div class=\"overview-content\">
        <h3>Database Health Overview</h3>
        <div class=\"status-overview\">
          ${this.renderHealthCard('Health Score', data.health_score || 85, 'fas fa-heartbeat')}
          ${this.renderMetricCard('Query Rate', data.queries_per_second || 0, 'qps', 'fas fa-tachometer-alt')}
          ${this.renderMetricCard('Avg Latency', data.avg_latency || 0, 'ms', 'fas fa-clock')}
          ${this.renderMetricCard('Connections', data.active_connections || 0, '', 'fas fa-plug')}
        </div>
        <div class=\"db-action-grid\">
          <div class=\"db-action-card\" onclick=\"window.dbManager.refreshOverview()\">
            <div class=\"db-action-card-header\">
              <div class=\"db-action-card-icon\"><i class=\"fas fa-sync-alt\"></i></div>
              <h4 class=\"db-action-card-title\">Refresh Data</h4>
            </div>
            <p class=\"db-action-card-description\">Update all overview metrics</p>
          </div>
        </div>
      </div>
    `;
  }

  renderRealtimeContent() {
    return `
      <div class=\"realtime-content\">
        <h3>Real-time Database Monitoring</h3>
        <div class=\"realtime-container\">
          <!-- Real-time components will be initialized here -->
        </div>
      </div>
    `;
  }

  renderTablesContent(data) {
    const tables = data.tables || [];
    return `
      <div class=\"tables-content\">
        <div class=\"tables-header\">
          <h3>Database Tables</h3>
          <div class=\"tables-controls\">
            <input type=\"search\" class=\"db-form-input\" placeholder=\"Search tables...\" 
                   aria-label=\"Search tables\" id=\"table-search\">
          </div>
        </div>
        <div class=\"db-table-container\">
          <table class=\"db-table\" role=\"table\" aria-label=\"Database tables\">
            <thead>
              <tr>
                <th scope=\"col\">Table Name</th>
                <th scope=\"col\" class=\"db-table-cell-numeric\">Row Count</th>
                <th scope=\"col\" class=\"db-table-cell-numeric\">Size</th>
                <th scope=\"col\">Status</th>
                <th scope=\"col\">Actions</th>
              </tr>
            </thead>
            <tbody>
              ${tables.map(table => this.renderTableRow(table)).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  renderTableRow(table) {
    const statusClass = table.health > 80 ? 'healthy' : table.health > 60 ? 'warning' : 'error';
    return `
      <tr>
        <td>
          <strong>${this.escapeHtml(table.name)}</strong>
          ${table.description ? `<br><small class=\"text-muted\">${this.escapeHtml(table.description)}</small>` : ''}
        </td>
        <td class=\"db-table-cell-numeric\">${table.row_count?.toLocaleString() || 'N/A'}</td>
        <td class=\"db-table-cell-numeric\">${table.size_human || 'N/A'}</td>
        <td>
          <span class=\"db-status-badge ${statusClass}\">
            <span class=\"db-status-dot\"></span>
            ${table.health > 80 ? 'Healthy' : table.health > 60 ? 'Warning' : 'Issues'}
          </span>
        </td>
        <td>
          <button class=\"db-button secondary\" onclick=\"window.dbManager.showTableDetails('${table.name}')\">
            Details
          </button>
        </td>
      </tr>
    `;
  }

  renderQueriesContent(data) {
    const queries = data.queries || [];
    return `
      <div class=\"queries-content\">
        <h3>Query Performance</h3>
        <div class=\"db-table-container\">
          <table class=\"db-table\" role=\"table\" aria-label=\"Slow queries\">
            <thead>
              <tr>
                <th scope=\"col\">Query</th>
                <th scope=\"col\" class=\"db-table-cell-numeric\">Duration</th>
                <th scope=\"col\" class=\"db-table-cell-numeric\">Calls</th>
                <th scope=\"col\">Last Seen</th>
              </tr>
            </thead>
            <tbody>
              ${queries.map(query => this.renderQueryRow(query)).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  renderQueryRow(query) {
    return `
      <tr>
        <td>
          <code class=\"query-snippet\" title=\"${this.escapeHtml(query.query)}\">${this.truncateQuery(query.query)}</code>
        </td>
        <td class=\"db-table-cell-numeric\">${query.avg_duration}ms</td>
        <td class=\"db-table-cell-numeric\">${query.calls}</td>
        <td>${query.last_seen || 'N/A'}</td>
      </tr>
    `;
  }

  renderMigrationsContent(migrationData, schemaData) {
    return `
      <div class=\"migrations-content\">
        <h3>Schema & Migrations</h3>
        <div class=\"migration-status\">
          <h4>Migration Status</h4>
          <p>Current: <code>${migrationData.current || 'Unknown'}</code></p>
          <p>Head: <code>${migrationData.head || 'Unknown'}</code></p>
          ${migrationData.pending?.length ? `<p class=\"warning\">Pending migrations: ${migrationData.pending.length}</p>` : ''}
        </div>
        ${schemaData.drift_detected ? `
          <div class=\"schema-drift warning\">
            <h4>⚠️ Schema Drift Detected</h4>
            <p>Schema differences found. Review recommended.</p>
          </div>
        ` : ''}
      </div>
    `;
  }

  renderLogsContent(data) {
    const logs = data.logs || [];
    return `
      <div class=\"logs-content\">
        <div class=\"logs-header\">
          <h3>Database Logs</h3>
          <div class=\"logs-controls\">
            <select class=\"db-form-select\" id=\"log-level-filter\" aria-label=\"Filter by log level\">
              <option value=\"\">All Levels</option>
              <option value=\"ERROR\">Error</option>
              <option value=\"WARN\">Warning</option>
              <option value=\"INFO\">Info</option>
              <option value=\"DEBUG\">Debug</option>
            </select>
          </div>
        </div>
        <div class=\"log-entries\" role=\"log\" aria-label=\"Database log entries\">
          ${logs.map(log => this.renderLogEntry(log)).join('')}
        </div>
      </div>
    `;
  }

  renderLogEntry(log) {
    const levelClass = log.level.toLowerCase();
    return `
      <div class=\"log-entry ${levelClass}\">
        <div class=\"log-meta\">
          <span class=\"log-timestamp\">${log.timestamp}</span>
          <span class=\"log-level db-status-badge ${levelClass}\">${log.level}</span>
        </div>
        <div class=\"log-message\">${this.escapeHtml(log.message)}</div>
      </div>
    `;
  }

  renderOperationsContent() {
    return `
      <div class=\"operations-content\">
        <h3>Database Operations</h3>
        <div class=\"operation-warning\">
          <p><strong>⚠️ Warning:</strong> These operations may impact database performance.</p>
        </div>
        <div class=\"db-action-grid\">
          ${this.renderOperationCard('vacuum', 'VACUUM Database', 'Clean up and optimize storage', 'fas fa-broom')}
          ${this.renderOperationCard('analyze', 'ANALYZE Tables', 'Update table statistics', 'fas fa-chart-bar')}
          ${this.renderOperationCard('reindex', 'REINDEX', 'Rebuild indexes', 'fas fa-wrench', 'warning')}
        </div>
      </div>
    `;
  }

  renderOperationCard(operation, title, description, icon, level = 'primary') {
    return `
      <div class=\"db-action-card ${level}\" onclick=\"window.dbManager.confirmOperation('${operation}')\">
        <div class=\"db-action-card-header\">
          <div class=\"db-action-card-icon\"><i class=\"${icon}\"></i></div>
          <h4 class=\"db-action-card-title\">${title}</h4>
        </div>
        <p class=\"db-action-card-description\">${description}</p>
      </div>
    `;
  }

  renderHealthCard(title, score, icon) {
    const level = score >= 80 ? 'healthy' : score >= 60 ? 'warning' : 'error';
    return `
      <div class=\"status-card\">
        <div class=\"status-card-header\">
          <div class=\"status-card-icon\"><i class=\"${icon}\"></i></div>
          <h4 class=\"status-card-title\">${title}</h4>
        </div>
        <div class=\"status-card-value\">${score}</div>
        <div class=\"status-card-change ${level}\">${level.charAt(0).toUpperCase() + level.slice(1)}</div>
      </div>
    `;
  }

  renderMetricCard(title, value, unit, icon) {
    return `
      <div class=\"status-card\">
        <div class=\"status-card-header\">
          <div class=\"status-card-icon\"><i class=\"${icon}\"></i></div>
          <h4 class=\"status-card-title\">${title}</h4>
        </div>
        <div class=\"status-card-value\">${typeof value === 'number' ? value.toLocaleString() : value}${unit ? ` ${unit}` : ''}</div>
      </div>
    `;
  }

  renderErrorContent(message) {
    return `
      <div class=\"error-content\" role=\"alert\">
        <div class=\"db-status-badge error\">
          <i class=\"fas fa-exclamation-triangle\"></i>
          Error
        </div>
        <p>${this.escapeHtml(message)}</p>
        <button class=\"db-button secondary\" onclick=\"location.reload()\">
          <i class=\"fas fa-refresh\"></i> Retry
        </button>
      </div>
    `;
  }

  initializeTableInteractions(panel) {
    const searchInput = panel.querySelector('#table-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.filterTables(e.target.value);
      });
    }
  }

  initializeQueryInteractions(panel) {
    // Add query interaction handlers
  }

  initializeLogInteractions(panel) {
    const levelFilter = panel.querySelector('#log-level-filter');
    if (levelFilter) {
      levelFilter.addEventListener('change', (e) => {
        this.filterLogs(e.target.value);
      });
    }
  }

  initializeOperationInteractions(panel) {
    // Operation interactions handled via onclick attributes for now
  }

  filterTables(searchTerm) {
    const rows = document.querySelectorAll('.db-table tbody tr');
    rows.forEach(row => {
      const tableName = row.querySelector('td strong').textContent;
      const matches = tableName.toLowerCase().includes(searchTerm.toLowerCase());
      row.style.display = matches ? '' : 'none';
    });
  }

  filterLogs(level) {
    const entries = document.querySelectorAll('.log-entry');
    entries.forEach(entry => {
      const entryLevel = entry.querySelector('.log-level').textContent;
      const matches = !level || entryLevel === level;
      entry.style.display = matches ? '' : 'none';
    });
  }

  setupResizeObserver() {
    if (typeof ResizeObserver !== 'undefined') {
      const resizeObserver = new ResizeObserver(entries => {
        // Handle responsive behavior
        this.handleResize();
      });
      resizeObserver.observe(this.container);
    }
  }

  handleResize() {
    // Update tab list scrolling if needed
    if (this.tabList.scrollWidth > this.tabList.clientWidth) {
      this.tabList.classList.add('scrollable');
    } else {
      this.tabList.classList.remove('scrollable');
    }
  }

  handleHashChange() {
    const hash = window.location.hash.slice(1);
    if (!hash) return;

    const tabIndex = Array.from(this.tabs).findIndex(tab => 
      tab.getAttribute('aria-controls') === hash
    );

    if (tabIndex !== -1) {
      this.activateTab(tabIndex);
    }
  }

  loadSavedTab() {
    try {
      const savedTab = localStorage.getItem('db-manager-active-tab');
      const hash = window.location.hash.slice(1);
      
      if (hash) {
        this.handleHashChange();
      } else if (savedTab) {
        const tabIndex = parseInt(savedTab, 10);
        if (tabIndex >= 0 && tabIndex < this.tabs.length) {
          this.activateTab(tabIndex);
          return;
        }
      }
      
      // Default to first tab
      this.activateTab(0);
    } catch (error) {
      console.warn('Failed to load saved tab state:', error);
      this.activateTab(0);
    }
  }

  saveTabState() {
    try {
      localStorage.setItem('db-manager-active-tab', this.currentTabIndex.toString());
    } catch (error) {
      console.warn('Failed to save tab state:', error);
    }
  }

  // Utility methods
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  truncateQuery(query, maxLength = 100) {
    if (query.length <= maxLength) return this.escapeHtml(query);
    return this.escapeHtml(query.slice(0, maxLength)) + '…';
  }

  // Public methods for global access
  refreshOverview() {
    const overviewPanel = document.querySelector('#overview-panel');
    if (overviewPanel) {
      overviewPanel.dataset.loaded = 'false';
      this.loadOverviewContent(overviewPanel);
    }
  }

  showTableDetails(tableName) {
    // Implementation for showing table details modal
    console.log('Show details for table:', tableName);
  }

  confirmOperation(operation) {
    const confirmed = confirm(`Are you sure you want to run ${operation.toUpperCase()}? This may impact performance.`);
    if (confirmed) {
      this.executeOperation(operation);
    }
  }

  async executeOperation(operation) {
    try {
      const response = await fetch(`/api/database/ops/${operation}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const result = await response.json();
      if (result.success) {
        this.showToast('success', `${operation.toUpperCase()} completed successfully`);
      } else {
        this.showToast('error', `${operation.toUpperCase()} failed: ${result.message}`);
      }
    } catch (error) {
      this.showToast('error', `${operation.toUpperCase()} failed: ${error.message}`);
    }
  }

  showToast(type, message) {
    if (window.showAlert) {
      window.showAlert(message, type);
    } else {
      console.log(`${type.toUpperCase()}: ${message}`);
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.dbManager = new DatabaseManagerTabs();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DatabaseManagerTabs;
}
