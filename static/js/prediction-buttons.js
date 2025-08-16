// Enhanced Prediction Buttons Module
// Handles all prediction-related button interactions with V3 endpoints

class PredictionButtonManager {
    constructor() {
        this.activeRequests = new Map();
        this.initializeButtons();
    }

    initializeButtons() {
        // Initialize all prediction buttons on the page
        document.addEventListener('click', (event) => {
            if (event.target.matches('.predict-btn, .btn-predict')) {
                event.preventDefault();
                this.handlePredictionClick(event.target);
            }
            
            if (event.target.matches('.run-batch-predictions')) {
                event.preventDefault();
                this.handleBatchPredictionClick(event.target);
            }
            
            if (event.target.matches('.run-all-predictions')) {
                event.preventDefault();
                this.handleRunAllPredictionsClick(event.target);
            }
        });
    }

    async handlePredictionClick(button) {
        const raceId = button.dataset.raceId;
        const raceFilename = button.dataset.raceFilename;
        
        if (!raceId && !raceFilename) {
            this.showErrorToast('Missing race identifier');
            return;
        }

        const requestKey = raceId || raceFilename;
        
        if (this.activeRequests.has(requestKey)) {
            this.showWarningToast('Prediction already in progress for this race');
            return;
        }

        await this.runSinglePrediction(button, raceId, raceFilename);
    }

    async handleBatchPredictionClick(button) {
        const selectedCheckboxes = document.querySelectorAll('.race-checkbox:checked');
        
        if (selectedCheckboxes.length === 0) {
            this.showWarningToast('Please select races to predict');
            return;
        }

        const races = Array.from(selectedCheckboxes).map(checkbox => ({
            raceId: checkbox.dataset.raceId,
            raceFilename: checkbox.dataset.raceFilename
        }));

        await this.runBatchPredictions(button, races);
    }

    async handleRunAllPredictionsClick(button) {
        await this.runAllUpcomingPredictions(button);
    }

    async runSinglePrediction(button, raceId, raceFilename) {
        const requestKey = raceId || raceFilename;
        const originalText = button.innerHTML;
        
        try {
            this.activeRequests.set(requestKey, true);
            this.setButtonLoading(button, 'Predicting...');
            
            const requestBody = {};
            if (raceFilename) {
                requestBody.race_filename = raceFilename;
            } else {
                requestBody.race_id = raceId;
            }

            const response = await fetch('/api/predict_single_race_enhanced', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.setButtonSuccess(button, 'Predicted!');
                this.showSuccessToast('Prediction completed successfully');
                this.displayPredictionResult(result);
                
                // Reset button after delay
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('btn-success');
                    button.classList.add('btn-primary');
                    button.disabled = false;
                }, 3000);
            } else {
                this.setButtonError(button, 'Failed');
                this.showErrorToast(`Prediction failed: ${result.message}`);
                
                // Reset button after delay
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('btn-danger');
                    button.classList.add('btn-primary');
                    button.disabled = false;
                }, 3000);
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.setButtonError(button, 'Error');
            this.showErrorToast(`Prediction error: ${error.message}`);
            
            // Reset button after delay
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('btn-danger');
                button.classList.add('btn-primary');
                button.disabled = false;
            }, 3000);
        } finally {
            this.activeRequests.delete(requestKey);
        }
    }

    async runBatchPredictions(button, races) {
        const originalText = button.innerHTML;
        
        try {
            this.setButtonLoading(button, `Predicting ${races.length} races...`);
            
            const results = [];
            let successCount = 0;
            
            for (let i = 0; i < races.length; i++) {
                const race = races[i];
                const progress = Math.round(((i + 1) / races.length) * 100);
                
                this.setButtonLoading(button, `Predicting... ${progress}%`);
                
                try {
                    const requestBody = {};
                    if (race.raceFilename) {
                        requestBody.race_filename = race.raceFilename;
                    } else {
                        requestBody.race_id = race.raceId;
                    }

                    const response = await fetch('/api/predict_single_race_enhanced', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestBody)
                    });

                    if (response.ok) {
                        const result = await response.json();
                        results.push(result);
                        if (result.success) {
                            successCount++;
                        }
                    } else {
                        results.push({
                            success: false,
                            message: `HTTP ${response.status}: ${response.statusText}`,
                            race_id: race.raceId,
                            race_filename: race.raceFilename
                        });
                    }
                } catch (error) {
                    results.push({
                        success: false,
                        message: error.message,
                        race_id: race.raceId,
                        race_filename: race.raceFilename
                    });
                }
            }
            
            this.setButtonSuccess(button, `Completed: ${successCount}/${races.length}`);
            this.showInfoToast(`Batch prediction completed: ${successCount}/${races.length} successful`);
            this.displayBatchResults(results);
            
            // Reset button after delay
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('btn-success');
                button.classList.add('btn-primary');
                button.disabled = false;
            }, 5000);
            
        } catch (error) {
            console.error('Batch prediction error:', error);
            this.setButtonError(button, 'Batch Failed');
            this.showErrorToast(`Batch prediction error: ${error.message}`);
            
            // Reset button after delay
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('btn-danger');
                button.classList.add('btn-primary');
                button.disabled = false;
            }, 3000);
        }
    }

    async runAllUpcomingPredictions(button) {
        const originalText = button.innerHTML;
        
        try {
            this.setButtonLoading(button, 'Running all predictions...');
            
            const response = await fetch('/api/predict_all_upcoming_races_enhanced', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.setButtonSuccess(button, `Completed: ${result.success_count}/${result.total_races}`);
                this.showSuccessToast(`All predictions completed: ${result.success_count}/${result.total_races} successful`);
                this.displayBatchResults(result.predictions || []);
            } else {
                this.setButtonError(button, 'Failed');
                this.showErrorToast(`All predictions failed: ${result.message}`);
            }
            
            // Reset button after delay
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('btn-success', 'btn-danger');
                button.classList.add('btn-primary');
                button.disabled = false;
            }, 5000);
            
        } catch (error) {
            console.error('All predictions error:', error);
            this.setButtonError(button, 'Error');
            this.showErrorToast(`All predictions error: ${error.message}`);
            
            // Reset button after delay
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('btn-danger');
                button.classList.add('btn-primary');
                button.disabled = false;
            }, 3000);
        }
    }

    setButtonLoading(button, text) {
        button.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${text}`;
        button.disabled = true;
        button.classList.remove('btn-primary', 'btn-success', 'btn-danger');
        button.classList.add('btn-secondary');
    }

    setButtonSuccess(button, text) {
        button.innerHTML = `<i class="fas fa-check"></i> ${text}`;
        button.disabled = true;
        button.classList.remove('btn-primary', 'btn-secondary', 'btn-danger');
        button.classList.add('btn-success');
    }

    setButtonError(button, text) {
        button.innerHTML = `<i class="fas fa-times"></i> ${text}`;
        button.disabled = true;
        button.classList.remove('btn-primary', 'btn-secondary', 'btn-success');
        button.classList.add('btn-danger');
    }

    displayPredictionResult(result) {
        const container = document.getElementById('prediction-results-container') || 
                         document.getElementById('predictionResultsContainer');
        
        if (!container) {
            console.warn('No prediction results container found');
            return;
        }

        container.style.display = 'block';
        
        const resultHTML = this.generateResultHTML(result);
        
        const resultsBody = container.querySelector('#prediction-results-body, .prediction-results-body') ||
                           container;
        
        if (resultsBody) {
            resultsBody.innerHTML = resultHTML;
        }
    }

    displayBatchResults(results) {
        const container = document.getElementById('prediction-results-container') || 
                         document.getElementById('predictionResultsContainer');
        
        if (!container) {
            console.warn('No prediction results container found');
            return;
        }

        container.style.display = 'block';
        
        let html = '<div class="batch-results">';
        results.forEach((result, index) => {
            html += this.generateResultHTML(result, index);
        });
        html += '</div>';
        
        const resultsBody = container.querySelector('#prediction-results-body, .prediction-results-body') ||
                           container;
        
        if (resultsBody) {
            resultsBody.innerHTML = html;
        }
    }

    generateResultHTML(result, index = 0) {
        if (result.success && result.predictions && result.predictions.length > 0) {
            const topPick = result.predictions[0];
            const winProb = topPick.final_score || topPick.win_probability || topPick.confidence || 0;
            const dogName = topPick.dog_name || topPick.name || 'Unknown';
            const raceInfo = result.race_filename || result.race_id || `Race ${index + 1}`;
            
            return `
                <div class="alert alert-success mb-3">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="alert-heading mb-1">
                                <i class="fas fa-trophy text-warning"></i> ${raceInfo}
                            </h6>
                            <p class="mb-1">
                                <strong>Top Pick:</strong> ${dogName} 
                                <span class="badge bg-success">${(winProb * 100).toFixed(1)}%</span>
                            </p>
                            <small class="text-muted">
                                <i class="fas fa-info-circle"></i> 
                                ${result.predictions.length} dogs analyzed
                                ${result.predictor_used ? ` | Predictor: ${result.predictor_used}` : ''}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        } else {
            const errorMessage = result.message || result.error || 'Unknown error occurred';
            const raceInfo = result.race_filename || result.race_id || `Race ${index + 1}`;
            
            return `
                <div class="alert alert-danger mb-3">
                    <h6 class="alert-heading">
                        <i class="fas fa-exclamation-triangle"></i> ${raceInfo}
                    </h6>
                    <p class="mb-0"><strong>Error:</strong> ${errorMessage}</p>
                </div>
            `;
        }
    }

    showSuccessToast(message) {
        this.showToast(message, 'success');
    }

    showErrorToast(message) {
        this.showToast(message, 'danger');
    }

    showWarningToast(message) {
        this.showToast(message, 'warning');
    }

    showInfoToast(message) {
        this.showToast(message, 'info');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
}

// Initialize the prediction button manager when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PredictionButtonManager();
});

// Export for use in other modules
window.PredictionButtonManager = PredictionButtonManager;
