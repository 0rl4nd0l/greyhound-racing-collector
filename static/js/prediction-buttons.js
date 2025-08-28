// Enhanced Prediction Buttons Module
// Handles all prediction-related button interactions with V3 endpoints

class PredictionButtonManager {
    constructor() {
        this.activeRequests = new Map();
        this.initializeButtons();
        this.setupTgrToggle();
    }

    initializeButtons() {
        // Initialize all prediction buttons on the page (robust event delegation)
        document.addEventListener('click', (event) => {
            // Handle single predict buttons regardless of where inside the button the user clicks
            const predictBtn = event.target.closest('.predict-btn, .btn-predict');
            if (predictBtn) {
                event.preventDefault();
                this.handlePredictionClick(predictBtn);
                return;
            }

            // Handle batch predictions
            const batchBtn = event.target.closest('.run-batch-predictions');
            if (batchBtn) {
                event.preventDefault();
                this.handleBatchPredictionClick(batchBtn);
                return;
            }

            // Handle run-all predictions
            const allBtn = event.target.closest('.run-all-predictions');
            if (allBtn) {
                event.preventDefault();
                this.handleRunAllPredictionsClick(allBtn);
                return;
            }

            // Details toggle in results rendered by this module
            const detailsBtn = event.target.closest('.pred-details-btn');
            if (detailsBtn) {
                event.preventDefault();
                this.handleDetailsClick(detailsBtn);
                return;
            }
        });
    }

    // Setup TGR features toggle UI and persistence
    setupTgrToggle() {
        try {
            const toggleId = 'tgr-features-toggle';
            if (document.getElementById(toggleId)) return;

            // Determine initial state from localStorage
            let enabled = false;
            try {
                const saved = String(localStorage.getItem('tgr_enabled') || '0');
                enabled = (saved === '1' || saved.toLowerCase() === 'true');
            } catch (e) {
                enabled = false;
            }

            // Create control
            const wrapper = document.createElement('div');
            wrapper.className = 'form-check form-switch tgr-toggle-control';
            wrapper.style.cssText = 'position:fixed; top:70px; right:20px; z-index:1030; background:rgba(255,255,255,0.92); border:1px solid rgba(0,0,0,0.1); border-radius:8px; padding:6px 10px;';
            wrapper.title = 'Toggle TheGreyhoundReview (TGR) DB features during prediction';

            const input = document.createElement('input');
            input.type = 'checkbox';
            input.className = 'form-check-input';
            input.id = toggleId;
            input.checked = enabled;

            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.setAttribute('for', toggleId);
            label.textContent = 'Enable TGR Features';

            wrapper.appendChild(input);
            wrapper.appendChild(label);

            // Prefer toast container to keep UI tidy; fallback to body
            const toastContainer = document.querySelector('.toast-container');
            if (toastContainer) {
                toastContainer.appendChild(wrapper);
            } else {
                document.body.appendChild(wrapper);
            }

            input.addEventListener('change', () => {
                try {
                    localStorage.setItem('tgr_enabled', input.checked ? '1' : '0');
                } catch (e) {}
                const msg = input.checked ? 'TGR features enabled' : 'TGR features disabled';
                try { this.showInfoToast(msg); } catch (e) {}
            });
        } catch (e) {
            // ignore UI errors
        }
    }

    getTgrEnabled() {
        try {
            const el = document.getElementById('tgr-features-toggle');
            if (el) return !!el.checked;
        } catch (e) {}
        try {
            const saved = String(localStorage.getItem('tgr_enabled') || '0');
            return (saved === '1' || saved.toLowerCase() === 'true');
        } catch (e) {
            return false;
        }
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
            // Include runtime TGR toggle from UI
            requestBody.tgr_enabled = this.getTgrEnabled();

            let response = await fetch('/api/predict_single_race_enhanced', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                // Attempt smart fallback when file is missing (404)
                let errText = '';
                let errJson = null;
                try { errText = await response.text(); } catch {}
                try { errJson = errText ? JSON.parse(errText) : null; } catch {}

                const isMissingFile = response.status === 404 && errJson && (errJson.error_type === 'file_not_found' || /file not found/i.test(errJson.message||''));
                if (isMissingFile) {
                    // Use available context to download then predict
                    const venue = button.dataset.venue || '';
                    const raceNumber = button.dataset.raceNumber || '';
                    const raceUrl = button.dataset.raceUrl || '';

                    if (raceUrl || (venue && raceNumber)) {
                        this.setButtonLoading(button, 'Downloading...');
                        try {
                            const dlResp = await fetch('/api/download_and_predict_race', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(raceUrl ? { race_url: raceUrl } : { venue, race_number: String(raceNumber).trim() })
                            });
                            const dlData = await dlResp.json();
                            if (!dlResp.ok || !dlData || dlData.success !== true) {
                                const msg = (dlData && (dlData.message || dlData.error)) || `HTTP ${dlResp.status}`;
                                throw new Error(`Download failed: ${msg}`);
                            }
                            // Now re-run prediction with the downloaded filename if available
                            const fname = dlData.filename || requestBody.race_filename || '';
                            if (fname) {
                                this.setButtonLoading(button, 'Predicting...');
                                const pr = await fetch('/api/predict_single_race_enhanced', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ race_filename: fname, tgr_enabled: this.getTgrEnabled() })
                                });
                                const prData = await pr.json();
                                if (!pr.ok || !prData || prData.success !== true) {
                                    const msg = (prData && (prData.message || prData.error)) || `HTTP ${pr.status}`;
                                    throw new Error(`Prediction failed: ${msg}`);
                                }
                                this.setButtonSuccess(button, 'Predicted!');
                                this.showSuccessToast('Download + Predict completed');
                                this.displayPredictionResult(prData);
                                setTimeout(() => {
                                    button.innerHTML = originalText;
                                    button.classList.remove('btn-success');
                                    button.classList.add('btn-primary');
                                    button.disabled = false;
                                }, 3000);
                                return; // handled fallback
                            }
                        } catch (fallbackErr) {
                            console.warn('Fallback download+predict failed', fallbackErr);
                            throw new Error(`HTTP ${response.status}: ${errText || response.statusText}`);
                        }
                    }
                }
                throw new Error(`HTTP ${response.status}: ${errText || response.statusText}`);
            }

            const result = await response.json();
            
            const isSuccess = !!(result && result.success === true);
            if (isSuccess) {
                const msg = (result.message && String(result.message).trim()) || 'Prediction completed successfully';
                const predUsed = result.predictor_used ? ` (Predictor: ${result.predictor_used})` : '';
                this.setButtonSuccess(button, 'Predicted!');
                this.showSuccessToast(`${msg}${predUsed}`);
                this.displayPredictionResult(result);
                
                // Reset button after delay
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('btn-success');
                    button.classList.add('btn-primary');
                    button.disabled = false;
                }, 3000);
            } else {
                const msg = (result && result.message ? String(result.message).trim() : '') || `Prediction failed with status ${response.status}`;
                this.setButtonError(button, 'Failed');
                this.showErrorToast(msg);
                
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
                    // Include runtime TGR toggle from UI
                    requestBody.tgr_enabled = this.getTgrEnabled();

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
                },
                body: JSON.stringify({ tgr_enabled: this.getTgrEnabled() })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.setButtonSuccess(button, `Completed: ${result.success_count}/${result.total_races}`);
                const msg = (result.message && String(result.message).trim()) || `All predictions completed: ${result.success_count}/${result.total_races} successful`;
                this.showSuccessToast(msg);
                this.displayBatchResults(result.predictions || []);
            } else {
                this.setButtonError(button, 'Failed');
                const msg = (result && result.message ? String(result.message).trim() : '') || 'All predictions failed';
                this.showErrorToast(msg);
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
        this._lastResults = [result]; // keep a handle for details lookup
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
            // Auto-expand details for the first (single) result if available
            try {
                const btn = resultsBody.querySelector('.pred-details-btn');
                if (btn) {
                    // Defer to allow DOM paint, then fetch details
                    setTimeout(() => this.handleDetailsClick(btn), 50);
                }
            } catch (_) {}
        }
    }

    displayBatchResults(results) {
        this._lastResults = Array.isArray(results) ? results : [];
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
            // Auto-expand details for the first result if available
            try {
                const btn = resultsBody.querySelector('.pred-details-btn');
                if (btn) {
                    setTimeout(() => this.handleDetailsClick(btn), 50);
                }
            } catch (_) {}
        }
    }

    // Compute a race name suitable for /api/prediction_detail
    _computeRaceNameForApi(result) {
        try {
            const nestedFilename = result && result.prediction && result.prediction.race_info && result.prediction.race_info.filename;
            const rawName = (result && (result.race_filename || (result.race_id || ''))) || (nestedFilename || '');
            return String(rawName)
                .replace(/\.csv$/i, '')
                .replace(/^prediction_/, '')
                .replace(/\.json$/i, '');
        } catch (e) {
            return '';
        }
    }

    async _fetchPredictionDetail(name) {
        const resp = await fetch(`/api/prediction_detail/${encodeURIComponent(String(name))}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    }

    async handleDetailsClick(button) {
        try {
            const targetId = button.getAttribute('data-target');
            const raceName = button.getAttribute('data-race');
            const container = document.getElementById(targetId);
            if (!container) return;

            const expanded = button.getAttribute('data-expanded') === 'true';
            if (expanded) {
                container.style.display = 'none';
                button.setAttribute('data-expanded', 'false');
                button.innerHTML = '<i class="fas fa-chevron-down"></i> Details';
                return;
            }

            container.style.display = 'block';
            container.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading details...</div>';

            const data = await this._fetchPredictionDetail(raceName);
            if (data && data.success && data.prediction) {
                const pred = data.prediction;
                const raceCtx = pred.race_context || {};
                const enhanced = pred.enhanced_predictions || pred.predictions || [];
                const items = Array.isArray(enhanced) ? enhanced.slice(0, 10) : [];

                let detailsHTML = '';
                detailsHTML += `
                    <div class="card mb-2">
                      <div class="card-body p-2">
                        <div class="small"><strong>Venue:</strong> ${raceCtx.venue || 'Unknown'} | <strong>Date:</strong> ${raceCtx.race_date || 'Unknown'} | <strong>Distance:</strong> ${raceCtx.distance || 'Unknown'} | <strong>Grade:</strong> ${raceCtx.grade || 'Unknown'}</div>
                      </div>
                    </div>`;

                if (items.length) {
                    detailsHTML += '<div class="row">';
                    items.forEach((d, idx) => {
                        const name = d.dog_name || d.clean_name || 'Unknown';
                        const score = Number(d.final_score || d.prediction_score || d.confidence || 0);
                        const box = d.box_number || d.box || 'N/A';
                        const extra = Array.isArray(d.key_factors) && d.key_factors.length ? `<div class=\"mt-1\"><small>${d.key_factors.slice(0,3).join(' • ')}</small></div>` : '';
                        detailsHTML += `
                          <div class="col-md-6 mb-2">
                            <div class="card card-sm">
                              <div class="card-body p-2">
                                <h6 class="card-title mb-1">${idx + 1}. ${name}</h6>
                                <p class="card-text mb-1"><small>Box: ${box} | Confidence: ${(score*100).toFixed(1)}%</small></p>
                                ${extra}
                              </div>
                            </div>
                          </div>`;
                    });
                    detailsHTML += '</div>';
                } else {
                    detailsHTML += '<p class="text-muted">No detailed runners available.</p>';
                }

                // Official results (if available) + DB fallback
                try {
                    let rendered = false;
                    const resultsAvailable = !!(pred.results_available || (Array.isArray(pred.actual_placings) && pred.actual_placings.length > 0));
                    if (resultsAvailable) {
                        const rawPlacings = Array.isArray(pred.actual_placings) ? pred.actual_placings.slice() : [];
                        const sortedPlacings = rawPlacings.sort((a,b) => {
                            const ap = Number(a.finish_position || a.position || 99);
                            const bp = Number(b.finish_position || b.position || 99);
                            return ap - bp;
                        });
                        const winner = (pred.race_results && pred.race_results.winner_name) ? pred.race_results.winner_name : (sortedPlacings[0] ? (sortedPlacings[0].dog_name || 'Unknown') : null);
                        const winnerOdds = pred.race_results && pred.race_results.winner_odds ? pred.race_results.winner_odds : null;
                        const winnerMargin = pred.race_results && pred.race_results.winner_margin ? pred.race_results.winner_margin : null;
                        const evalBlock = (pred.evaluation && (pred.evaluation.winner_predicted || pred.evaluation.top3_hit))
                            ? `<div class=\"mt-1\">${pred.evaluation.winner_predicted ? '<span class=\"badge bg-success me-1\"><i class=\"fas fa-check\"></i> Winner predicted</span>' : ''}${pred.evaluation.top3_hit ? '<span class=\"badge bg-info\">Top 3 hit</span>' : ''}</div>`
                            : '';

                        let listHTML = '';
                        if (sortedPlacings.length) {
                            listHTML = '<ol class="mb-0 ps-3">' + sortedPlacings.map((p) => {
                                const pos = p.finish_position || p.position || '?';
                                const nm = p.dog_name || 'Unknown';
                                const bx = (p.box_number !== undefined && p.box_number !== null) ? ` (Box ${p.box_number})` : '';
                                const t = (p.individual_time ? ` — ${p.individual_time}s` : '');
                                const m = (p.margin ? `, ${p.margin}` : '');
                                return `<li><strong>${nm}</strong>${bx}${t}${m}</li>`;
                            }).join('') + '</ol>';
                        }

                        detailsHTML += `
                          <div class="card mt-2">
                            <div class="card-header p-2 d-flex justify-content-between align-items-center">
                              <div><i class="fas fa-flag-checkered"></i> Official Results</div>
                              ${evalBlock}
                            </div>
                            <div class="card-body p-2">
                              ${winner ? `<div class=\"mb-2\"><strong>Winner:</strong> ${winner}${winnerOdds ? ` (Odds: ${winnerOdds})` : ''}${winnerMargin ? `, Margin: ${winnerMargin}` : ''}</div>` : ''}
                              ${listHTML || '<div class=\"text-muted\">No placings available</div>'}
                            </div>
                          </div>`;
                        rendered = true;
                    }
                    if (!rendered) {
                        // Fallback: query DB via /api/races/results using race_context keys
                        try {
                            const ctx = pred.race_context || pred.race_info || {};
                            const venue = ctx.venue || '';
                            const date = ctx.race_date || ctx.date || '';
                            const raceNum = ctx.race_number || '';
                            if (venue && date && raceNum) {
                                const qs = new URLSearchParams({ venue: String(venue), date: String(date), race_number: String(raceNum) });
                                const res = await fetchWithErrorHandling(`/api/races/results?${qs.toString()}`);
                                const data = await res.json();
                                if (data && data.success && Array.isArray(data.results) && data.results.length) {
                                    const sortedPlacings = data.results.slice().sort((a,b) => Number(a.finish_position||a.position||99) - Number(b.finish_position||b.position||99));
                                    const winner = data.winner_name || (sortedPlacings[0] ? (sortedPlacings[0].dog_name || 'Unknown') : null);
                                    const winnerOdds = data.winner_odds || null;
                                    const winnerMargin = data.winner_margin || null;
                                    const evalBlock = (pred.evaluation && (pred.evaluation.winner_predicted || pred.evaluation.top3_hit))
                                        ? `<div class=\"mt-1\">${pred.evaluation.winner_predicted ? '<span class=\"badge bg-success me-1\"><i class=\"fas fa-check\"></i> Winner predicted</span>' : ''}${pred.evaluation.top3_hit ? '<span class=\"badge bg-info\">Top 3 hit</span>' : ''}</div>`
                                        : '';
                                    const listHTML = '<ol class="mb-0 ps-3">' + sortedPlacings.map((p) => {
                                        const nm = p.dog_name || 'Unknown';
                                        const bx = (p.box_number !== undefined && p.box_number !== null) ? ` (Box ${p.box_number})` : '';
                                        const t = (p.individual_time ? ` — ${p.individual_time}s` : '');
                                        const m = (p.margin ? `, ${p.margin}` : '');
                                        return `<li><strong>${nm}</strong>${bx}${t}${m}</li>`;
                                    }).join('') + '</ol>';
                                    detailsHTML += `
                                      <div class="card mt-2">
                                        <div class="card-header p-2 d-flex justify-content-between align-items-center">
                                          <div><i class="fas fa-flag-checkered"></i> Official Results</div>
                                          ${evalBlock}
                                        </div>
                                        <div class="card-body p-2">
                                          ${winner ? `<div class=\"mb-2\"><strong>Winner:</strong> ${winner}${winnerOdds ? ` (Odds: ${winnerOdds})` : ''}${winnerMargin ? `, Margin: ${winnerMargin}` : ''}</div>` : ''}
                                          ${listHTML || '<div class=\"text-muted\">No placings available</div>'}
                                        </div>
                                      </div>`;
                                    rendered = true;
                                }
                            }
                        } catch (err) { console.warn('fallback results fetch failed', err); }
                    }
                } catch (e) { console.warn('results render failed', e); }

                container.innerHTML = detailsHTML;
            } else {
                container.innerHTML = `<div class="alert alert-warning">Could not load prediction details for ${raceName}</div>`;
            }

            button.setAttribute('data-expanded', 'true');
            const icon = button.querySelector('i');
            if (icon) icon.className = 'fas fa-chevron-up';
            button.innerHTML = '<i class="fas fa-chevron-up"></i> Hide';
        } catch (e) {
            const targetId = button.getAttribute('data-target');
            const container = document.getElementById(targetId);
            if (container) container.innerHTML = `<div class="alert alert-danger">Error loading details: ${e.message}</div>`;
        }
    }

    generateResultHTML(result, index = 0) {
        // Normalize predictions (support nested shape from enhanced endpoint)
        const predictions = Array.isArray(result.predictions)
            ? result.predictions
            : (result.prediction && Array.isArray(result.prediction.predictions) ? result.prediction.predictions : []);
        const msgTop = (result && (result.message || result.error)) || '';
        const msgNested = (result && result.prediction && (result.prediction.message || '')) || '';
        const rawMsg = `${String(msgTop)} ${String(msgNested)}`.trim();
        const completion = /prediction\s+completed/i.test(rawMsg);

        const effectiveSuccess = !!result.success || (Array.isArray(predictions) && predictions.length > 0) || completion;
        const raceInfo = result.race_filename || result.race_id || `Race ${index + 1}`;
        const raceNameForApi = this._computeRaceNameForApi(result);

        if (effectiveSuccess) {
            if (Array.isArray(predictions) && predictions.length > 0) {
                const topPick = predictions[0];
                const winProb = topPick.final_score || topPick.win_probability || topPick.confidence || 0;
                const dogName = topPick.dog_name || topPick.name || 'Unknown';
                const total = predictions.length;
                return `
                    <div class="alert ${result.degraded ? 'alert-warning' : 'alert-success'} mb-3">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="alert-heading mb-1">
                                    <i class="fas fa-trophy text-warning"></i> ${raceInfo}
                                </h6>
                                <p class="mb-1">
                                    <strong>Top Pick:</strong> ${dogName}
                                    <span class="badge bg-success">${(Number(winProb) * 100).toFixed(1)}%</span>
                                </p>
                                <small class="text-muted">
                                    <i class="fas fa-info-circle"></i>
                                    ${total} dogs analyzed
                                    ${result.predictor_used ? ` | Predictor: ${result.predictor_used}` : ''}
                                    ${raceNameForApi ? ` | <a class="link-secondary" href="/api/prediction_detail/${encodeURIComponent(raceNameForApi)}" target="_blank" rel="noopener noreferrer"><i class="fas fa-file-code"></i> Raw JSON</a>` : ''}
                                </small>
                            </div>
                            <button class="btn btn-sm btn-outline-primary pred-details-btn" data-race="${raceNameForApi}" data-target="pb-details-${index}" data-expanded="false">
                                <i class="fas fa-chevron-down"></i> Details
                            </button>
                        </div>
                        <div class="prediction-details" id="pb-details-${index}" style="display:none; margin-top: 10px;"></div>
                    </div>
                `;
            }
            // Success-shaped with no predictions payload
            const message = rawMsg || 'Prediction completed';
            return `
                <div class="alert ${result.degraded ? 'alert-warning' : 'alert-success'} mb-3">
                    <h6 class="alert-heading">
                        <i class="fas fa-check-circle"></i> ${raceInfo}
                    </h6>
                    <p class="mb-1">${message}</p>
                    <small class="text-muted">
                        ${result.predictor_used ? `Predictor: ${result.predictor_used}` : ''}
                        ${raceNameForApi ? ` | <a class=\"link-secondary\" href=\"/api/prediction_detail/${encodeURIComponent(raceNameForApi)}\" target=\"_blank\" rel=\"noopener noreferrer\"><i class=\"fas fa-file-code\"></i> Raw JSON</a>` : ''}
                    </small>
                    ${raceNameForApi ? `<div class=\"mt-2\"><button class=\"btn btn-sm btn-outline-primary pred-details-btn\" data-race=\"${raceNameForApi}\" data-target=\"pb-details-${index}\" data-expanded=\"false\"><i class=\"fas fa-chevron-down\"></i> Details</button></div>` : ''}
                    <div class="prediction-details" id="pb-details-${index}" style="display:none; margin-top: 10px;"></div>
                </div>
            `;
        }

        const errorMessage = rawMsg || 'Unknown error occurred';
        return `
            <div class="alert alert-danger mb-3">
                <h6 class="alert-heading">
                    <i class="fas fa-exclamation-triangle"></i> ${raceInfo}
                </h6>
                <p class="mb-0">${errorMessage}</p>
            </div>
        `;
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
