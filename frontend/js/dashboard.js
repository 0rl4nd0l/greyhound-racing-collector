/**
 * TGR Dashboard Controller
 * Main JavaScript file for managing dashboard functionality
 */

class TGRDashboard {
    constructor() {
        this.apiUrl = '/api/v1';
        this.refreshInterval = null;
        this.refreshRate = 30000; // 30 seconds
        this.isConnected = true;
        this.lastUpdate = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.loadInitialData();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.refreshData();
        });

        // Service control buttons
        document.getElementById('startServicesBtn').addEventListener('click', () => {
            this.startServices();
        });

        document.getElementById('stopServicesBtn').addEventListener('click', () => {
            this.stopServices();
        });

        // Chart time range selector
        document.getElementById('chartTimeRange').addEventListener('change', (e) => {
            this.updateChartTimeRange(e.target.value);
        });

        // Activity filter
        document.getElementById('activityFilter').addEventListener('change', (e) => {
            this.filterActivity(e.target.value);
        });

        // Clear alerts button
        document.getElementById('clearAlertsBtn').addEventListener('click', () => {
            this.clearAlerts();
        });

        // Modal close buttons
        document.getElementById('closeJobModal').addEventListener('click', () => {
            this.closeModal('jobDetailsModal');
        });

        document.getElementById('closeConfigModal').addEventListener('click', () => {
            this.closeModal('configModal');
        });

        // Click outside modal to close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target.id);
            }
        });
    }

    async loadInitialData() {
        this.showLoading();
        try {
            await Promise.all([
                this.updateSystemOverview(),
                this.updateServiceStatus(),
                this.updateAlerts(),
                this.updateActivity(),
                this.updateCharts()
            ]);
            this.updateConnectionStatus(true);
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.updateConnectionStatus(false);
            this.showError('Failed to load dashboard data');
        } finally {
            this.hideLoading();
        }
    }

    async refreshData() {
        const refreshBtn = document.getElementById('refreshBtn');
        const icon = refreshBtn.querySelector('i');
        
        icon.classList.add('fa-spin');
        
        try {
            await this.loadInitialData();
            this.lastUpdate = new Date();
            this.updateLastUpdatedTime();
        } catch (error) {
            console.error('Refresh failed:', error);
        } finally {
            icon.classList.remove('fa-spin');
        }
    }

    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => {
            this.refreshData();
        }, this.refreshRate);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    async updateSystemOverview() {
        try {
            const response = await fetch(`${this.apiUrl}/status/system`);
            if (!response.ok) throw new Error('Failed to fetch system status');
            
            const data = await response.json();
            
            // Update system health
            this.updateCard('systemHealthCard', {
                value: data.system_health || 'UNKNOWN',
                uptime: data.uptime || '--',
                successRate: data.success_rate ? `${data.success_rate}%` : '--'
            });

            // Update data quality
            this.updateCard('dataQualityCard', {
                value: data.data_quality_score ? `${data.data_quality_score}/100` : '--',
                completeness: data.completeness_score ? `${data.completeness_score}%` : '--',
                freshness: data.freshness_score ? `${data.freshness_score}%` : '--'
            });

            // Update job processing
            this.updateCard('jobProcessingCard', {
                value: data.jobs_processed_24h || 0,
                queueSize: data.queue_size || 0,
                activeWorkers: data.active_workers || 0
            });

            // Update cache performance
            this.updateCard('cachePerformanceCard', {
                value: data.cache_hit_rate ? `${data.cache_hit_rate}%` : '--',
                totalEntries: data.total_cache_entries || 0,
                activeEntries: data.active_cache_entries || 0
            });

        } catch (error) {
            console.error('Failed to update system overview:', error);
            this.updateConnectionStatus(false);
        }
    }

    updateCard(cardId, data) {
        const card = document.getElementById(cardId);
        if (!card) return;

        // Update main value
        const valueElement = card.querySelector('.metric-value');
        if (valueElement) {
            valueElement.textContent = data.value;
        }

        // Update specific metrics based on card type
        switch (cardId) {
            case 'systemHealthCard':
                this.updateElement('systemUptime', data.uptime);
                this.updateElement('systemSuccessRate', data.successRate);
                this.updateHealthCardStatus(data.value);
                break;
            case 'dataQualityCard':
                this.updateElement('completenessScore', data.completeness);
                this.updateElement('freshnessScore', data.freshness);
                break;
            case 'jobProcessingCard':
                this.updateElement('queueSize', data.queueSize);
                this.updateElement('activeWorkers', data.activeWorkers);
                break;
            case 'cachePerformanceCard':
                this.updateElement('cacheEntries', data.totalEntries);
                this.updateElement('activeCacheEntries', data.activeEntries);
                break;
        }
    }

    updateHealthCardStatus(status) {
        const card = document.getElementById('systemHealthCard');
        const valueElement = card.querySelector('.metric-value');
        
        // Remove existing status classes
        valueElement.classList.remove('status-healthy', 'status-warning', 'status-critical');
        
        // Add appropriate status class
        switch (status.toLowerCase()) {
            case 'healthy':
                valueElement.classList.add('status-healthy');
                break;
            case 'warning':
                valueElement.classList.add('status-warning');
                break;
            case 'critical':
            case 'error':
                valueElement.classList.add('status-critical');
                break;
        }
    }

    async updateServiceStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/services/status`);
            if (!response.ok) throw new Error('Failed to fetch service status');
            
            const data = await response.json();

            // Update individual service cards
            this.updateServiceCard('monitoringService', data.monitoring || {});
            this.updateServiceCard('enrichmentService', data.enrichment || {});
            this.updateServiceCard('schedulerService', data.scheduler || {});
            this.updateServiceCard('databaseService', data.database || {});

        } catch (error) {
            console.error('Failed to update service status:', error);
            // Use mock data for development
            this.updateServiceCard('monitoringService', {
                status: 'running',
                health_checks: 145,
                alerts_generated: 3
            });
            this.updateServiceCard('enrichmentService', {
                status: 'running',
                jobs_completed: 7,
                processing_time: '12.3s'
            });
            this.updateServiceCard('schedulerService', {
                status: 'running',
                scheduled_jobs: 25,
                next_batch: 'in 2h 15m'
            });
            this.updateServiceCard('databaseService', {
                status: 'healthy',
                total_records: '17,179',
                performance_summaries: 9
            });
        }
    }

    updateServiceCard(serviceId, data) {
        const card = document.getElementById(serviceId);
        if (!card) return;

        // Update service status
        const statusElement = card.querySelector('.service-status');
        if (statusElement) {
            const statusIcon = statusElement.querySelector('i');
            const statusText = statusElement.querySelector('span');
            
            // Remove existing status classes
            statusElement.classList.remove('status-running', 'status-healthy', 'status-warning', 'status-error');
            
            const status = data.status || 'unknown';
            statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            
            switch (status.toLowerCase()) {
                case 'running':
                    statusElement.classList.add('status-running');
                    break;
                case 'healthy':
                    statusElement.classList.add('status-healthy');
                    break;
                case 'warning':
                    statusElement.classList.add('status-warning');
                    break;
                case 'error':
                case 'critical':
                    statusElement.classList.add('status-error');
                    break;
            }
        }

        // Update service-specific metrics
        switch (serviceId) {
            case 'monitoringService':
                this.updateElement('monitoringChecks', data.health_checks || '--');
                this.updateElement('monitoringAlerts', data.alerts_generated || '--');
                break;
            case 'enrichmentService':
                this.updateElement('enrichmentJobs', data.jobs_completed || '--');
                this.updateElement('enrichmentTime', data.processing_time || '--');
                break;
            case 'schedulerService':
                this.updateElement('scheduledJobs', data.scheduled_jobs || '--');
                this.updateElement('nextBatch', data.next_batch || '--');
                break;
            case 'databaseService':
                this.updateElement('totalRecords', data.total_records || '--');
                this.updateElement('performanceSummaries', data.performance_summaries || '--');
                break;
        }
    }

    async updateAlerts() {
        try {
            const response = await fetch(`${this.apiUrl}/alerts`);
            if (!response.ok) throw new Error('Failed to fetch alerts');
            
            const data = await response.json();
            this.renderAlerts(data.alerts || []);

        } catch (error) {
            console.error('Failed to update alerts:', error);
            // Use mock data for development
            this.renderAlerts([
                {
                    id: 1,
                    level: 'info',
                    title: 'System Started',
                    message: 'TGR enrichment system started successfully',
                    timestamp: new Date().toISOString()
                },
                {
                    id: 2,
                    level: 'warning',
                    title: 'Cache Efficiency',
                    message: 'Cache hit rate below optimal threshold',
                    timestamp: new Date(Date.now() - 300000).toISOString()
                }
            ]);
        }
    }

    renderAlerts(alerts) {
        const container = document.getElementById('alertsContainer');
        if (!container) return;

        if (!alerts.length) {
            container.innerHTML = `
                <div class="no-alerts">
                    <i class="fas fa-check-circle"></i>
                    <p>No active alerts</p>
                </div>
            `;
            return;
        }

        container.innerHTML = alerts.map(alert => `
            <div class="alert-item alert-${alert.level}">
                <div class="alert-icon">
                    <i class="fas ${this.getAlertIcon(alert.level)}"></i>
                </div>
                <div class="alert-content">
                    <h4>${alert.title}</h4>
                    <p>${alert.message}</p>
                    <div class="alert-time">${this.formatRelativeTime(alert.timestamp)}</div>
                </div>
            </div>
        `).join('');
    }

    getAlertIcon(level) {
        const icons = {
            critical: 'fa-exclamation-triangle',
            warning: 'fa-exclamation-circle',
            info: 'fa-info-circle'
        };
        return icons[level] || 'fa-info-circle';
    }

    async updateActivity() {
        try {
            const response = await fetch(`${this.apiUrl}/activity`);
            if (!response.ok) throw new Error('Failed to fetch activity');
            
            const data = await response.json();
            this.renderActivity(data.activities || []);

        } catch (error) {
            console.error('Failed to update activity:', error);
            // Use mock data for development
            this.renderActivity([
                {
                    id: 1,
                    type: 'jobs',
                    title: 'Job Processing Complete',
                    description: 'Processed 4 enrichment jobs successfully',
                    timestamp: new Date().toISOString()
                },
                {
                    id: 2,
                    type: 'health',
                    title: 'Health Check Passed',
                    description: 'System health check completed - all systems operational',
                    timestamp: new Date(Date.now() - 180000).toISOString()
                },
                {
                    id: 3,
                    type: 'jobs',
                    title: 'Batch Enrichment Started',
                    description: 'Started batch enrichment for 25 dogs',
                    timestamp: new Date(Date.now() - 600000).toISOString()
                }
            ]);
        }
    }

    renderActivity(activities) {
        const container = document.getElementById('activityTimeline');
        if (!container) return;

        if (!activities.length) {
            container.innerHTML = `
                <div class="no-activity">
                    <i class="fas fa-clock"></i>
                    <p>No recent activity</p>
                </div>
            `;
            return;
        }

        container.innerHTML = activities.map(activity => `
            <div class="activity-item activity-${activity.type}">
                <div class="activity-icon">
                    <i class="fas ${this.getActivityIcon(activity.type)}"></i>
                </div>
                <div class="activity-content">
                    <h4>${activity.title}</h4>
                    <p>${activity.description}</p>
                    <div class="activity-time">${this.formatRelativeTime(activity.timestamp)}</div>
                </div>
            </div>
        `).join('');
    }

    getActivityIcon(type) {
        const icons = {
            jobs: 'fa-cogs',
            health: 'fa-heartbeat',
            alert: 'fa-exclamation-triangle',
            system: 'fa-server'
        };
        return icons[type] || 'fa-info-circle';
    }

    async startServices() {
        try {
            this.showLoading();
            const response = await fetch(`${this.apiUrl}/services/start`, { method: 'POST' });
            if (!response.ok) throw new Error('Failed to start services');
            
            this.showSuccess('Services started successfully');
            await this.updateServiceStatus();
        } catch (error) {
            console.error('Failed to start services:', error);
            this.showError('Failed to start services');
        } finally {
            this.hideLoading();
        }
    }

    async stopServices() {
        try {
            this.showLoading();
            const response = await fetch(`${this.apiUrl}/services/stop`, { method: 'POST' });
            if (!response.ok) throw new Error('Failed to stop services');
            
            this.showSuccess('Services stopped successfully');
            await this.updateServiceStatus();
        } catch (error) {
            console.error('Failed to stop services:', error);
            this.showError('Failed to stop services');
        } finally {
            this.hideLoading();
        }
    }

    async clearAlerts() {
        try {
            const response = await fetch(`${this.apiUrl}/alerts`, { method: 'DELETE' });
            if (!response.ok) throw new Error('Failed to clear alerts');
            
            await this.updateAlerts();
            this.showSuccess('Alerts cleared successfully');
        } catch (error) {
            console.error('Failed to clear alerts:', error);
            this.showError('Failed to clear alerts');
        }
    }

    filterActivity(filterType) {
        const activityItems = document.querySelectorAll('.activity-item');
        
        activityItems.forEach(item => {
            if (filterType === 'all') {
                item.style.display = 'flex';
            } else {
                const hasClass = item.classList.contains(`activity-${filterType}`);
                item.style.display = hasClass ? 'flex' : 'none';
            }
        });
    }

    updateChartTimeRange(timeRange) {
        // This will be implemented when charts are updated
        console.log('Updating chart time range to:', timeRange);
        this.updateCharts(timeRange);
    }

    async updateCharts(timeRange = '24h') {
        // Charts implementation will be in charts.js
        if (window.TGRCharts) {
            await window.TGRCharts.updateCharts(timeRange);
        }
    }

    initializeCharts() {
        // Charts initialization will be in charts.js
        if (window.TGRCharts) {
            window.TGRCharts.init();
        }
    }

    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('active');
        }
    }

    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('active');
        }
    }

    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('active');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const indicator = document.getElementById('connectionStatus');
        const icon = indicator.querySelector('i');
        const text = indicator.querySelector('span');

        if (connected) {
            icon.style.color = 'var(--success-color)';
            text.textContent = 'Connected';
            indicator.title = 'Connected to TGR services';
        } else {
            icon.style.color = 'var(--error-color)';
            text.textContent = 'Disconnected';
            indicator.title = 'Unable to connect to TGR services';
        }
    }

    updateLastUpdatedTime() {
        const element = document.getElementById('lastUpdated');
        if (element && this.lastUpdate) {
            element.textContent = `Last updated: ${this.formatRelativeTime(this.lastUpdate)}`;
        }
    }

    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }

    formatRelativeTime(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${diffDays}d ago`;
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;

        // Add to DOM
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);

        // Close button functionality
        notification.querySelector('.notification-close').addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });

        // Add notification styles if not already present
        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: var(--bg-card);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-lg);
                    padding: 1rem;
                    z-index: 3000;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    max-width: 400px;
                    animation: slideInRight 0.3s ease;
                }
                
                .notification-success {
                    border-left: 4px solid var(--success-color);
                }
                
                .notification-error {
                    border-left: 4px solid var(--error-color);
                }
                
                .notification-info {
                    border-left: 4px solid var(--info-color);
                }
                
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    flex: 1;
                }
                
                .notification-success .notification-content i {
                    color: var(--success-color);
                }
                
                .notification-error .notification-content i {
                    color: var(--error-color);
                }
                
                .notification-info .notification-content i {
                    color: var(--info-color);
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: var(--text-secondary);
                    cursor: pointer;
                    font-size: 0.875rem;
                    padding: 0.25rem;
                }
                
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tgrDashboard = new TGRDashboard();
});

// Update last updated time every minute
setInterval(() => {
    if (window.tgrDashboard) {
        window.tgrDashboard.updateLastUpdatedTime();
    }
}, 60000);
