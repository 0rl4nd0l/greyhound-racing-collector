/**
 * TGR Dashboard Charts Module
 * ==========================
 * 
 * Simple charts module using Chart.js for the TGR dashboard.
 */

class TGRCharts {
    constructor() {
        this.performanceChart = null;
        this.jobStatusChart = null;
        this.initialized = false;
    }

    /**
     * Initialize all charts
     */
    async initializeCharts() {
        try {
            console.log('📊 Initializing charts...');
            
            // Wait for Chart.js to be available
            if (typeof Chart === 'undefined') {
                console.error('Chart.js not loaded');
                return false;
            }

            // Initialize performance trends chart
            this.initializePerformanceChart();
            
            // Initialize job status chart
            this.initializeJobStatusChart();

            this.initialized = true;
            console.log('✅ Charts initialized successfully');
            return true;

        } catch (error) {
            console.error('Failed to initialize charts:', error);
            return false;
        }
    }

    /**
     * Initialize performance trends line chart
     */
    initializePerformanceChart() {
        const canvas = document.getElementById('performanceChart');
        if (!canvas) {
            console.warn('Performance chart canvas not found');
            return;
        }

        const ctx = canvas.getContext('2d');
        
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Jobs Processed',
                        data: [],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Success Rate (%)',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Cache Hit Rate (%)',
                        data: [],
                        borderColor: '#17a2b8',
                        backgroundColor: 'rgba(23, 162, 184, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Jobs'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        },
                        min: 0,
                        max: 100,
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    /**
     * Initialize job status doughnut chart
     */
    initializeJobStatusChart() {
        const canvas = document.getElementById('jobStatusChart');
        if (!canvas) {
            console.warn('Job status chart canvas not found');
            return;
        }

        const ctx = canvas.getContext('2d');
        
        this.jobStatusChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Completed', 'Pending', 'Running', 'Failed'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        '#28a745', // Completed - green
                        '#ffc107', // Pending - yellow
                        '#007bff', // Running - blue
                        '#dc3545'  // Failed - red
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                    }
                }
            }
        });
    }

    /**
     * Update performance chart with new data
     */
    updatePerformanceChart(data) {
        if (!this.performanceChart) {
            console.warn('Performance chart not initialized');
            return;
        }

        try {
            this.performanceChart.data.labels = data.labels;
            this.performanceChart.data.datasets[0].data = data.jobsProcessed;
            this.performanceChart.data.datasets[1].data = data.successRate;
            this.performanceChart.data.datasets[2].data = data.cacheHitRate;
            this.performanceChart.update();
        } catch (error) {
            console.error('Error updating performance chart:', error);
        }
    }

    /**
     * Update job status chart with new data
     */
    updateJobStatusChart(data) {
        if (!this.jobStatusChart) {
            console.warn('Job status chart not initialized');
            return;
        }

        try {
            this.jobStatusChart.data.datasets[0].data = [
                data.completed || 0,
                data.pending || 0,
                data.running || 0,
                data.failed || 0
            ];
            this.jobStatusChart.update();
        } catch (error) {
            console.error('Error updating job status chart:', error);
        }
    }

    /**
     * Update charts with sample data for demo purposes
     */
    updateWithSampleData() {
        console.log('📊 Loading sample chart data...');
        
        // Sample performance data
        const now = new Date();
        const labels = [];
        const jobsProcessed = [];
        const successRate = [];
        const cacheHitRate = [];

        for (let i = 23; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60 * 60 * 1000);
            labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
            jobsProcessed.push(Math.floor(Math.random() * 8) + 2);
            successRate.push(95 + Math.random() * 5);
            cacheHitRate.push(85 + Math.random() * 15);
        }

        this.updatePerformanceChart({
            labels,
            jobsProcessed,
            successRate,
            cacheHitRate
        });

        // Sample job status data
        this.updateJobStatusChart({
            completed: 7,
            pending: 2,
            running: 1,
            failed: 0
        });
    }

    /**
     * Resize charts (useful for responsive behavior)
     */
    resizeCharts() {
        if (this.performanceChart) {
            this.performanceChart.resize();
        }
        if (this.jobStatusChart) {
            this.jobStatusChart.resize();
        }
    }

    /**
     * Destroy charts (cleanup)
     */
    destroy() {
        if (this.performanceChart) {
            this.performanceChart.destroy();
            this.performanceChart = null;
        }
        if (this.jobStatusChart) {
            this.jobStatusChart.destroy();
            this.jobStatusChart = null;
        }
        this.initialized = false;
    }
}

// Create global instance
window.TGRCharts = new TGRCharts();

// Handle window resize
window.addEventListener('resize', () => {
    if (window.TGRCharts && window.TGRCharts.initialized) {
        window.TGRCharts.resizeCharts();
    }
});
