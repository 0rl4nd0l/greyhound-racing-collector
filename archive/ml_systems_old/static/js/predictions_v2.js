
document.addEventListener('DOMContentLoaded', () => {
    const raceCheckboxes = document.querySelectorAll('.race-checkbox');
    const bulkActionsSelect = document.getElementById('bulk-actions');
    const runBulkActionsBtn = document.getElementById('run-bulk-actions');
    const exportCsvBtn = document.getElementById('export-csv');
    const exportPdfBtn = document.getElementById('export-pdf');
    const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
    const autoRefreshInterval = document.getElementById('auto-refresh-interval');
    const runAllPredictionsBtn = document.getElementById('run-all-predictions');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    // Advisory elements
    const autoAdvisoryToggle = document.getElementById('auto-advisory-toggle');
    const generateAdvisoryBtn = document.getElementById('generate-advisory-btn');
    const advisoryMessagesContainer = document.getElementById('advisory-messages-predict');

    let autoRefreshTimer;
    let lastPredictionData = null;
    let currentFilePath = null;
    let processingStatusTimer;
    
    // Initialize processing status monitoring
    initializeProcessingStatusMonitoring();

function getSelectedRaceIds() {
        const selectedIds = [];
        // Use Array.from to ensure forEach is available on all browsers
        Array.from(raceCheckboxes).forEach(checkbox = {
            if (checkbox.checked) {
                selectedIds.push(checkbox.dataset.raceId);
            }
        });
        return selectedIds;
    }
    
    function storePredictionData(predictionData, filePath) {
        lastPredictionData = predictionData;
        currentFilePath = filePath;
    }

async function runBulkPredictions(raceIds) {
    let completed = 0;
    const total = raceIds.length;
    
    // Use Array.from to ensure forEach is available on all browsers
    for (const raceId of raceIds) {
        // Simulate prediction run
        await new Promise(resolve => setTimeout(resolve, Math.random() * 2000));
        completed++;
        const progress = (completed / total) * 100;
        progressBar.style.width = `${progress}%`;
        progressText.innerText = `Processed ${completed} of ${total}`;
        if (completed === total) {
progressText.innerText = 'Completed all predictions.';
            
            // Storing last prediction data (if available)
            const predictionData = getLatestPredictionData();
            storePredictionData(predictionData, null);

            if (autoAdvisoryToggle.checked) {
                await generateAdvisory(lastPredictionData, currentFilePath);
            }
        }
    }
}

    function exportToCsv() {
        const data = [['Race ID', 'Dog', 'Win Probability']];
        // Populate data from the table
        const tableRows = document.querySelectorAll('#predictions-table tbody tr');
        // Use Array.from to ensure forEach is available on all browsers
        Array.from(tableRows).forEach(row => {
            const raceId = row.cells[0].innerText;
            const dog = row.cells[1].innerText;
            const winProb = row.cells[2].innerText;
            data.push([raceId, dog, winProb]);
        });

        const csv = Papa.unparse(data);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', 'predictions.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function exportToPdf() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        doc.autoTable({ html: '#predictions-table' });
        doc.save('predictions.pdf');
    }

    function handleAutoRefresh() {
        if (autoRefreshToggle.checked) {
            const interval = parseInt(autoRefreshInterval.value, 10) * 1000;
            localStorage.setItem('autoRefreshInterval', interval);
            autoRefreshTimer = setInterval(() => {
                // Add logic to refresh predictions data
                console.log('Auto-refreshing predictions...');
            }, interval);
        } else {
            clearInterval(autoRefreshTimer);
        }
        localStorage.setItem('autoRefreshEnabled', autoRefreshToggle.checked);
    }

    runBulkActionsBtn.addEventListener('click', () => {
        const action = bulkActionsSelect.value;
        const selectedIds = getSelectedRaceIds();

        if (action === 'run-predictions' && selectedIds.length > 0) {
            runBulkPredictions(selectedIds);
        }
    });

    if(exportCsvBtn) {
        exportCsvBtn.addEventListener('click', exportToCsv);
    }

    if(exportPdfBtn) {
        exportPdfBtn.addEventListener('click', exportToPdf);
    }
    
autoRefreshToggle.addEventListener('change', handleAutoRefresh);
    autoRefreshInterval.addEventListener('change', handleAutoRefresh);

    if (generateAdvisoryBtn) {
        generateAdvisoryBtn.addEventListener('click', async () => {
            await generateAdvisoryManually();
        });
    }

    if(runAllPredictionsBtn) {
        runAllPredictionsBtn.addEventListener('click', () => {
            const allRaceIds = Array.from(raceCheckboxes).map(cb => cb.dataset.raceId);
            runBulkPredictions(allRaceIds);
        });
    }

    // Load saved auto-refresh settings
    const savedInterval = localStorage.getItem('autoRefreshInterval');
    const savedEnabled = localStorage.getItem('autoRefreshEnabled');

    if (savedInterval) {
        autoRefreshInterval.value = savedInterval / 1000;
    }
    if (savedEnabled === 'true') {
        autoRefreshToggle.checked = true;
        handleAutoRefresh();
    }

function generateAdvisoryManually() {
        if (lastPredictionData) {
            generateAdvisory(lastPredictionData, currentFilePath);
        } else {
            const predictionData = getLatestPredictionData();
            storePredictionData(predictionData, null);
            generateAdvisory(predictionData, currentFilePath);
        }
    }

    function getLatestPredictionData() {
        // Mock: Get the latest prediction data
        // Replace this with actual logic to get prediction data from state or API
        return {
            prediction: 'sample data',
        };
    }

    async function generateAdvisory(predictionData, filePath = null) {
        if (!advisoryMessagesContainer) {
            console.warn('Advisory messages container not found');
            return;
        }

        // Show loading spinner
        if (window.AdvisoryUtils && window.AdvisoryUtils.showAdvisoryLoading) {
            window.AdvisoryUtils.showAdvisoryLoading(
                advisoryMessagesContainer,
                'Generating AI advisory...'
            );
        } else {
            advisoryMessagesContainer.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Generating AI advisory...</div>';
        }

        // Prepare payload
        const payload = {};
        if (filePath) {
            payload.file_path = filePath;
        } else {
            payload.prediction_data = predictionData;
        }

        try {
            // Make API call to generate advisory
            const response = await fetch('/api/generate_advisory', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.success) {
                // Clear loading state and render advisory
                if (window.AdvisoryUtils && window.AdvisoryUtils.clearAdvisory) {
                    window.AdvisoryUtils.clearAdvisory(advisoryMessagesContainer);
                }
                
                if (window.AdvisoryUtils && window.AdvisoryUtils.renderAdvisory) {
                    window.AdvisoryUtils.renderAdvisory(result, advisoryMessagesContainer);
                } else {
                    // Fallback rendering if AdvisoryUtils not available
                    renderAdvisoryFallback(result);
                }
            } else {
                throw new Error(result.message || 'Advisory generation failed');
            }
        } catch (error) {
            console.error('Error generating advisory:', error);
            
            // Clear loading state
            if (window.AdvisoryUtils && window.AdvisoryUtils.clearAdvisory) {
                window.AdvisoryUtils.clearAdvisory(advisoryMessagesContainer);
            }
            
            // Show error alert
            showAlert(`Advisory generation failed: ${error.message}`, 'danger', advisoryMessagesContainer);
        }
    }
    
    // Fallback advisory rendering if AdvisoryUtils is not available
    function renderAdvisoryFallback(result) {
        const alertClass = result.type === 'success' ? 'alert-success' :
                          result.type === 'warning' ? 'alert-warning' :
                          result.type === 'danger' ? 'alert-danger' : 'alert-info';
        
        const html = `
            <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                <h6><i class="fas fa-lightbulb"></i> ${result.title || 'Advisory'}</h6>
                <p>${result.message || 'Advisory generated successfully'}</p>
                ${result.details && result.details.length > 0 ?
                    `<ul class="mb-0">${result.details.map(detail => `<li>${detail}</li>`).join('')}</ul>` :
                    ''
                }
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        advisoryMessagesContainer.innerHTML = html;
    }
    
    // Enhanced alert function that can target specific containers
    function showAlert(message, type, container = null) {
        // Use the enhanced alert logic for containers
        if (container) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            container.innerHTML = '';
            container.appendChild(alertDiv);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
            return;
        }
        
        // Global alert fallback - check if global showAlert exists
        if (typeof window.showAlert === 'function' && window.showAlert !== showAlert) {
            window.showAlert(message, type);
            return;
        }
        
        // Final fallback - create floating alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 1050; max-width: 400px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    // Processing status monitoring function
    function initializeProcessingStatusMonitoring() {
        const statusElement = document.getElementById('processing-status');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const progressContainer = document.querySelector('.progress-container') || document.querySelector('.progress');
        
        function updateProcessingStatus(data) {
            if (!data) return;
            
            if (data.running) {
                // Update status text
                if (statusElement) {
                    statusElement.textContent = data.current_task || 'Processing...';
                }
                
                // Update progress bar
                if (progressBar) {
                    progressBar.style.width = data.progress + '%';
                    progressBar.textContent = data.progress + '%';
                    
                    // Update progress bar color based on progress
                    progressBar.className = 'progress-bar';
                    if (data.progress === 100) {
                        progressBar.classList.add('bg-success');
                    } else if (data.progress > 0) {
                        progressBar.classList.add('bg-primary');
                    }
                }
                
                // Update progress text
                if (progressText) {
                    progressText.textContent = `${data.progress}% - ${data.current_task || 'Processing'}`;
                }
                
                // Show enhanced CSV scraper info
                if (data.current_task && data.current_task.includes('CSV')) {
                    showCSVScraperInfo(progressContainer);
                }
                
                // Show processing logs if available
                if (data.log && data.log.length > 0) {
                    updateProcessingLogs(data.log);
                }
                
                // Continue polling while running
                if (processingStatusTimer) {
                    clearTimeout(processingStatusTimer);
                }
                processingStatusTimer = setTimeout(fetchProcessingStatus, 2000);
                
            } else {
                // Reset UI when not running
                if (statusElement) {
                    statusElement.textContent = 'Idle';
                }
                if (progressBar) {
                    progressBar.style.width = '0%';
                    progressBar.textContent = '0%';
                    progressBar.className = 'progress-bar';
                }
                if (progressText) {
                    progressText.textContent = 'Ready';
                }
                
                // Clean up CSV scraper info
                hideCSVScraperInfo();
                
                // Stop polling
                if (processingStatusTimer) {
                    clearTimeout(processingStatusTimer);
                    processingStatusTimer = null;
                }
            }
        }
        
        function showCSVScraperInfo(container) {
            if (!container) return;
            
            let infoDiv = document.querySelector('.csv-progress-info');
            if (!infoDiv) {
                infoDiv = document.createElement('div');
                infoDiv.className = 'csv-progress-info mt-2';
                infoDiv.innerHTML = `
                    <div class="card border-info">
                        <div class="card-body py-2">
                            <h6 class="card-title mb-1"><i class="fas fa-download"></i> CSV Scraper Status</h6>
                            <div class="row text-sm">
                                <div class="col-6">
                                    <small class="text-muted">Workers:</small> <span class="badge bg-info">2 parallel</span>
                                </div>
                                <div class="col-6">
                                    <small class="text-muted">Timeout:</small> <span class="badge bg-warning">5 minutes</span>
                                </div>
                            </div>
                            <div class="mt-1">
                                <small class="text-muted">Mode:</small> <span class="badge bg-success">Expert form scraping</span>
                                <small class="text-muted ml-2">Days ahead:</small> <span class="badge bg-primary">1 day</span>
                            </div>
                        </div>
                    </div>
                `;
                container.appendChild(infoDiv);
            }
        }
        
        function hideCSVScraperInfo() {
            const csvInfo = document.querySelector('.csv-progress-info');
            if (csvInfo) {
                csvInfo.remove();
            }
        }
        
        function updateProcessingLogs(logs) {
            const logsContainer = document.getElementById('processing-logs');
            if (!logsContainer) return;
            
            // Clear existing logs and add new ones
            logsContainer.innerHTML = '';
            
            // Show last 10 log entries
            const recentLogs = logs.slice(-10);
            recentLogs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = 'mb-1 p-1 rounded';
                
                // Style based on log level
                let logClass = 'text-info';
                let icon = 'fas fa-info-circle';
                
                if (log.level === 'ERROR' || (log.message && log.message.includes('❌'))) {
                    logClass = 'text-danger bg-danger bg-opacity-10';
                    icon = 'fas fa-exclamation-triangle';
                } else if (log.level === 'WARNING' || (log.message && log.message.includes('⚠️'))) {
                    logClass = 'text-warning bg-warning bg-opacity-10';
                    icon = 'fas fa-exclamation-circle';
                } else if (log.message && log.message.includes('✅')) {
                    logClass = 'text-success bg-success bg-opacity-10';
                    icon = 'fas fa-check-circle';
                } else if (log.message && log.message.includes('🚀')) {
                    logClass = 'text-primary bg-primary bg-opacity-10';
                    icon = 'fas fa-rocket';
                }
                
                logEntry.className += ` ${logClass}`;
                
                const timestamp = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '';
                logEntry.innerHTML = `
                    <small class="d-flex align-items-start">
                        <i class="${icon} me-1 mt-1 flex-shrink-0"></i>
                        <span class="flex-grow-1">${log.message}</span>
                        ${timestamp ? `<span class="text-muted ms-2 flex-shrink-0">${timestamp}</span>` : ''}
                    </small>
                `;
                
                logsContainer.appendChild(logEntry);
            });
            
            // Auto-scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        function fetchProcessingStatus() {
            fetch('/api/processing_status')
                .then(response => response.json())
                .then(data => {
                    updateProcessingStatus(data);
                })
                .catch(error => {
                    console.error('Error fetching processing status:', error);
                    // Stop polling on error
                    if (processingStatusTimer) {
                        clearTimeout(processingStatusTimer);
                        processingStatusTimer = null;
                    }
                });
        }
        
        // Start initial fetch
        fetchProcessingStatus();
        
        // Expose function globally for manual triggering
        window.fetchProcessingStatus = fetchProcessingStatus;
    }
    
    // Expose functions globally for template access
    window.generateAdvisoryForPredictionV2 = generateAdvisory;
    window.generateAdvisoryManuallyV2 = generateAdvisoryManually;
});
