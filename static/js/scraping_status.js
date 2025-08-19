/**
 * Enhanced Processing Status Monitor
 * Provides comprehensive progress tracking for CSV scraping and other background tasks
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize both standalone and integrated progress tracking
    if (typeof window.initializeProcessingStatusMonitoring !== 'function') {
        initializeProgressTracking();
    }
});

// Standalone progress tracking for pages without predictions_v2.js
function initializeProgressTracking() {
    const statusElement = document.getElementById('processing-status');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressContainer = document.querySelector('.progress-container') || document.querySelector('.progress');
    const logsContainer = document.getElementById('processing-logs');
    
    let pollingInterval = null;
    let isPolling = false;

    function updateProgressStatus(data) {
        if (!data) return;
        
        if (data.running) {
            // Update status elements
            if (statusElement) {
                statusElement.textContent = data.current_task || 'Processing...';
            }
            
            if (progressBar) {
                progressBar.style.width = data.progress + '%';
                progressBar.setAttribute('aria-valuenow', data.progress);
                
                // Update progress bar appearance
                progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
                if (data.progress === 100) {
                    progressBar.classList.add('bg-success');
                    progressBar.classList.remove('progress-bar-animated');
                } else if (data.progress > 0) {
                    progressBar.classList.add('bg-primary');
                }
                
                // Show percentage text in progress bar
                progressBar.textContent = data.progress + '%';
            }
            
            if (progressText) {
                const taskText = data.current_task || 'Processing';
                const progressPercent = data.progress || 0;
                progressText.textContent = `${progressPercent}% - ${taskText}`;
            }

            // Show enhanced CSV scraper information
            if (data.current_task && data.current_task.toLowerCase().includes('csv')) {
                showEnhancedCSVInfo(progressContainer, data);
            }
            
            // Update processing logs
            if (data.log && data.log.length > 0) {
                updateProcessingLogs(data.log, logsContainer);
            }

            // Continue polling while running
            if (!isPolling) {
                startPolling();
            }
            
        } else {
            // Reset UI when processing is complete or idle
            resetProgressUI();
            stopPolling();
        }
    }
    
    function showEnhancedCSVInfo(container, data) {
        if (!container) return;
        
        let infoDiv = document.querySelector('.csv-progress-info');
        if (!infoDiv) {
            infoDiv = document.createElement('div');
            infoDiv.className = 'csv-progress-info mt-3';
            infoDiv.innerHTML = `
                <div class="card border-info shadow-sm">
                    <div class="card-header bg-info bg-opacity-10 py-2">
                        <h6 class="card-title mb-0">
                            <i class="fas fa-download text-info"></i> 
                            CSV Scraper Status
                        </h6>
                    </div>
                    <div class="card-body py-2">
                        <div class="row g-2 text-sm">
                            <div class="col-6">
                                <div class="d-flex justify-content-between">
                                    <small class="text-muted">Workers:</small> 
                                    <span class="badge bg-info">2 parallel</span>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="d-flex justify-content-between">
                                    <small class="text-muted">Timeout:</small> 
                                    <span class="badge bg-warning text-dark">5 minutes</span>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="d-flex justify-content-between">
                                    <small class="text-muted">Mode:</small> 
                                    <span class="badge bg-success">Expert form</span>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="d-flex justify-content-between">
                                    <small class="text-muted">Days ahead:</small> 
                                    <span class="badge bg-primary">1 day</span>
                                </div>
                            </div>
                        </div>
                        <div class="mt-2 pt-2 border-top">
                            <small class="text-muted d-block">
                                <i class="fas fa-info-circle"></i> 
                                Fetching CSV form guides with concurrent processing for faster results
                            </small>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(infoDiv);
        }
        
        // Update dynamic info if available
        updateCSVInfoDynamics(infoDiv, data);
    }
    
    function updateCSVInfoDynamics(infoDiv, data) {
        // Add dynamic status updates based on current progress
        const dynamicStatus = infoDiv.querySelector('.dynamic-status');
        if (!dynamicStatus) {
            const statusDiv = document.createElement('div');
            statusDiv.className = 'dynamic-status mt-1';
            infoDiv.querySelector('.card-body').appendChild(statusDiv);
        }
        
        let statusText = '';
        let statusClass = 'text-info';
        
        if (data.progress < 25) {
            statusText = 'Initializing scraper...';
            statusClass = 'text-info';
        } else if (data.progress < 50) {
            statusText = 'Preparing expert-form method...';
            statusClass = 'text-warning';
        } else if (data.progress < 90) {
            statusText = 'Running parallel CSV fetching...';
            statusClass = 'text-primary';
        } else if (data.progress < 100) {
            statusText = 'Processing results...';
            statusClass = 'text-success';
        } else {
            statusText = 'Completed successfully!';
            statusClass = 'text-success';
        }
        
        const statusEl = infoDiv.querySelector('.dynamic-status');
        if (statusEl) {
            statusEl.innerHTML = `<small class="${statusClass}"><i class="fas fa-circle fa-xs"></i> ${statusText}</small>`;
        }
    }
    
    function updateProcessingLogs(logs, container) {
        if (!container) return;
        
        // Clear and rebuild logs
        container.innerHTML = '';
        
        // Show last 15 log entries with enhanced styling
        const recentLogs = logs.slice(-15);
        
        recentLogs.forEach((log, index) => {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry mb-1 p-2 rounded border-start border-3';
            
            // Style based on log content and level
            let logClass = 'border-info bg-light';
            let icon = 'fas fa-info-circle text-info';
            let timestamp = '';
            
            if (log.timestamp) {
                timestamp = new Date(log.timestamp).toLocaleTimeString();
            }
            
            // Determine styling based on log content
            if (log.level === 'ERROR' || (log.message && log.message.includes('❌'))) {
                logClass = 'border-danger bg-danger bg-opacity-10';
                icon = 'fas fa-exclamation-triangle text-danger';
            } else if (log.level === 'WARNING' || (log.message && log.message.includes('⚠️'))) {
                logClass = 'border-warning bg-warning bg-opacity-10';
                icon = 'fas fa-exclamation-circle text-warning';
            } else if (log.message && log.message.includes('✅')) {
                logClass = 'border-success bg-success bg-opacity-10';
                icon = 'fas fa-check-circle text-success';
            } else if (log.message && log.message.includes('🚀')) {
                logClass = 'border-primary bg-primary bg-opacity-10';
                icon = 'fas fa-rocket text-primary';
            }
            
            logEntry.className += ` ${logClass}`;
            
            logEntry.innerHTML = `
                <div class="d-flex align-items-start">
                    <i class="${icon} me-2 mt-1 flex-shrink-0"></i>
                    <div class="flex-grow-1">
                        <small class="fw-medium">${log.message}</small>
                        ${timestamp ? `<small class="text-muted d-block mt-1">${timestamp}</small>` : ''}
                    </div>
                </div>
            `;
            
            container.appendChild(logEntry);
        });
        
        // Auto-scroll to bottom
        container.scrollTop = container.scrollHeight;
    }
    
    function resetProgressUI() {
        if (statusElement) {
            statusElement.textContent = 'Idle';
        }
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            progressBar.className = 'progress-bar';
            progressBar.setAttribute('aria-valuenow', 0);
        }
        if (progressText) {
            progressText.textContent = 'Ready';
        }
        
        // Clean up CSV info
        const csvInfo = document.querySelector('.csv-progress-info');
        if (csvInfo) {
            csvInfo.remove();
        }
    }
    
    function startPolling() {
        if (isPolling) return;
        
        isPolling = true;
        pollingInterval = setInterval(fetchProcessingStatus, 2000);
    }
    
    function stopPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
        isPolling = false;
    }

    function fetchProcessingStatus() {
        fetch('/api/processing_status')
            .then(response => response.json())
            .then(data => {
                updateProgressStatus(data);
            })
            .catch(error => {
                console.error('Error fetching processing status:', error);
                stopPolling();
            });
    }

    // Initial fetch
    fetchProcessingStatus();
    
    // Expose functions globally
    window.fetchProcessingStatus = fetchProcessingStatus;
    window.stopProcessingStatusPolling = stopPolling;
    window.startProcessingStatusPolling = startPolling;
}
