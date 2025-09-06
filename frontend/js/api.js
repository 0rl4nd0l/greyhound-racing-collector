/**
 * TGR Dashboard API Client
 * =======================
 * 
 * Simplified API client for communicating with the TGR dashboard backend.
 * Handles all REST API calls and basic WebSocket connections.
 */

class TGRApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.timeout = 10000; // 10 seconds
        this.socket = null;
        this.eventListeners = new Map();
    }

    /**
     * Make HTTP request with timeout and error handling
     */
    async makeRequest(url, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(this.baseUrl + url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }
            
            console.error(`API Error (${url}):`, error);
            throw error;
        }
    }

    /**
     * System Status APIs
     */
    async getSystemStatus() {
        return this.makeRequest('/api/v1/status/system');
    }

    async getServicesStatus() {
        return this.makeRequest('/api/v1/services/status');
    }

    async startServices() {
        return this.makeRequest('/api/v1/services/start', {
            method: 'POST'
        });
    }

    async stopServices() {
        return this.makeRequest('/api/v1/services/stop', {
            method: 'POST'
        });
    }

    /**
     * Alerts APIs
     */
    async getAlerts() {
        return this.makeRequest('/api/v1/alerts');
    }

    async clearAlerts() {
        return this.makeRequest('/api/v1/alerts', {
            method: 'DELETE'
        });
    }

    /**
     * Activity APIs
     */
    async getActivity() {
        return this.makeRequest('/api/v1/activity');
    }

    /**
     * Metrics APIs
     */
    async getPerformanceMetrics(timeRange = '24h') {
        return this.makeRequest(`/api/v1/metrics/performance?range=${timeRange}`);
    }

    /**
     * Jobs APIs
     */
    async createJob(dogName, jobType = 'comprehensive', priority = 5) {
        return this.makeRequest('/api/v1/jobs', {
            method: 'POST',
            body: JSON.stringify({
                dog_name: dogName,
                job_type: jobType,
                priority: priority
            })
        });
    }

    /**
     * Health Check
     */
    async healthCheck() {
        return this.makeRequest('/health');
    }

    /**
     * WebSocket Connection (simplified)
     */
    initializeWebSocket() {
        if (typeof io !== 'undefined') {
            try {
                this.socket = io();
                
                this.socket.on('connect', () => {
                    console.log('WebSocket connected');
                    this.emit('connect');
                });
                
                this.socket.on('disconnect', () => {
                    console.log('WebSocket disconnected');
                    this.emit('disconnect');
                });
                
                // Forward all events to local listeners
                ['status_update', 'service_status_change', 'job_created', 'alerts_cleared'].forEach(event => {
                    this.socket.on(event, (data) => {
                        this.emit(event, data);
                    });
                });
                
                return this.socket;
            } catch (error) {
                console.warn('WebSocket initialization failed:', error);
            }
        } else {
            console.warn('Socket.IO not available');
        }
        return null;
    }

    /**
     * Event handling
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }

    emit(event, data = null) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Mock data fallbacks
     */
    getMockSystemStatus() {
        return {
            system_health: 'HEALTHY',
            uptime: '2h 15m',
            success_rate: 100,
            data_quality_score: 95.2,
            completeness_score: 94.8,
            freshness_score: 95.6,
            jobs_processed_24h: 7,
            queue_size: 0,
            active_workers: 1,
            cache_hit_rate: 87.3,
            total_cache_entries: 1247,
            active_cache_entries: 1089
        };
    }

    getMockServicesStatus() {
        return {
            monitoring: {
                status: 'running',
                health_checks: 145,
                alerts_generated: 3,
                uptime: '2h 15m'
            },
            enrichment: {
                status: 'stopped',
                jobs_completed: 1247,
                processing_time: '12.3s',
                queue_size: 0,
                active_workers: 1
            },
            scheduler: {
                status: 'stopped',
                scheduled_jobs: 25,
                next_batch: 'in 2h 15m',
                last_run: '5m ago'
            },
            database: {
                status: 'healthy',
                total_records: '45,231',
                performance_summaries: 2847,
                connection_pool: '5/10'
            }
        };
    }

    getMockAlerts() {
        return {
            alerts: [
                {
                    id: 1,
                    level: 'info',
                    title: 'System Started',
                    message: 'TGR enrichment system started successfully',
                    timestamp: new Date().toISOString(),
                    acknowledged: false
                },
                {
                    id: 2,
                    level: 'warning',
                    title: 'Data Freshness',
                    message: 'Some data entries are older than 24 hours',
                    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
                    acknowledged: false
                }
            ]
        };
    }

    getMockActivity() {
        return {
            activities: [
                {
                    id: 1,
                    type: 'health',
                    title: 'Health Check Passed',
                    description: 'System health check completed - all systems operational',
                    timestamp: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
                    metadata: { health_score: 100 }
                },
                {
                    id: 2,
                    type: 'system',
                    title: 'Cache Refresh',
                    description: 'Feature cache refreshed for 25 dogs',
                    timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
                    metadata: { cache_entries: 25 }
                }
            ]
        };
    }

    getMockPerformanceMetrics() {
        const now = new Date();
        const labels = [];
        const jobsProcessed = [];
        const successRate = [];
        const cacheHitRate = [];

        // Generate 24 hours of sample data
        for (let i = 23; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60 * 60 * 1000);
            labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
            jobsProcessed.push(Math.floor(Math.random() * 10) + 3);
            successRate.push(95 + Math.random() * 5);
            cacheHitRate.push(85 + Math.random() * 15);
        }

        return {
            labels,
            datasets: {
                jobsProcessed,
                successRate,
                cacheHitRate
            },
            jobStatus: {
                completed: 7,
                pending: 2,
                running: 1,
                failed: 0
            }
        };
    }

    /**
     * API health check
     */
    async checkApiHealth() {
        try {
            await this.healthCheck();
            return true;
        } catch (error) {
            console.warn('API health check failed:', error);
            return false;
        }
    }
}

// Create global instance
window.TGRApiClient = new TGRApiClient();

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🔗 Initializing TGR API Client...');
    
    // Check if API is available
    const apiHealthy = await window.TGRApiClient.checkApiHealth();
    
    if (apiHealthy) {
        console.log('✅ API connection established');
        
        // Initialize WebSocket if available
        window.TGRApiClient.initializeWebSocket();
    } else {
        console.warn('⚠️ API not available - using mock data');
    }
});
