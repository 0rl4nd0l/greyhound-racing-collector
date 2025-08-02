// Interactive Races Page JavaScript

// CSRF Token helper function
function getCSRFToken() {
    const metaTag = document.querySelector('meta[name="csrf-token"]');
    return metaTag ? metaTag.getAttribute('content') : null;
}

// Enhanced fetch wrapper with CSRF and error handling
async function fetchWithErrorHandling(url, options = {}) {
    try {
        const defaultHeaders = {
            'Content-Type': 'application/json'
        };
        
        // Add CSRF token if available
        const csrfToken = getCSRFToken();
        if (csrfToken && (options.method === 'POST' || options.method === 'PUT' || options.method === 'DELETE')) {
            defaultHeaders['X-CSRFToken'] = csrfToken;
        }
        
        const finalOptions = {
            ...options,
            headers: {
                ...defaultHeaders,
                ...options.headers
            }
        };
        
        const response = await fetch(url, finalOptions);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
        }
        
        return response;
    } catch (error) {
        console.error(`Fetch error for ${url}:`, error);
        throw error;
    }
}

// Toast notification function
function showToast(message, type = 'info') {
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

document.addEventListener('DOMContentLoaded', () => {
    const state = {
        races: [],
        currentPage: 1,
        racesPerPage: 10,
        searchQuery: '',
        sortOrder: 'race_date|desc',
        filters: {},
        isLoading: false,
        viewMode: 'regular', // 'regular' or 'upcoming'
    };

    const elements = {
        searchBox: document.getElementById('search-races'),
        searchButton: document.getElementById('search-button'),
        selectAllCheckbox: document.getElementById('select-all-races'),
        racesTableBody: document.getElementById('races-table-body'),
        paginationControls: document.getElementById('pagination-controls'),
        runSelectedButton: document.getElementById('run-selected-predictions'),
        runAllUpcomingButton: document.getElementById('run-all-upcoming-predictions'),
        predictionResultsContainer: document.getElementById('prediction-results-container'),
        predictionResultsBody: document.getElementById('prediction-results-body'),
        viewToggleButton: document.getElementById('view-toggle-button'),
        viewModeLabel: document.getElementById('view-mode-label'),
    };

    // Initialize the page
    async function init() {
        try {
            await loadRaces();
            setupEventListeners();
            renderRaces();
        } catch (error) {
            console.error('Initialization failed:', error);
            showToast('Failed to initialize page. Please refresh and try again.', 'danger');
        }
    }

    // Load races from API with proper error handling
    async function loadRaces() {
        if (state.isLoading) return;
        
        state.isLoading = true;
        
        try {
            // Choose endpoint based on view mode
            const endpoint = state.viewMode === 'upcoming' ? '/api/upcoming_races_csv' : '/api/races/paginated';
            const response = await fetchWithErrorHandling(endpoint);
            const data = await response.json();
            
            if (data.success) {
                state.races = Array.isArray(data.races) ? data.races : [];
                const viewLabel = state.viewMode === 'upcoming' ? 'upcoming' : 'regular';
                showToast(`Successfully loaded ${state.races.length} ${viewLabel} races`, 'success');
            } else {
                throw new Error(data.message || 'Failed to load races');
            }
        } catch (error) {
            console.error('Failed to load races:', error);
            showToast(`Error loading races: ${error.message}`, 'danger');
            state.races = [];
        } finally {
            state.isLoading = false;
        }
    }

    // Render races in the table
    function renderRaces() {
        const filteredRaces = filterAndSortRaces();
        const paginatedRaces = paginateRaces(filteredRaces);

        elements.racesTableBody.innerHTML = '';
        if (paginatedRaces.length === 0) {
            elements.racesTableBody.innerHTML = '<tr><td colspan="8" class="text-center">No races found.</td></tr>';
            return;
        }

        paginatedRaces.forEach(race => {
            const row = document.createElement('tr');
            
            // Add race_filename data attribute for upcoming races
            const raceFilenamAttr = race.filename ? `data-race-filename="${race.filename}"` : '';
            
            row.innerHTML = `
                <td><input type="checkbox" class="race-checkbox" data-race-id="${race.race_id}" ${raceFilenamAttr}></td>
                <td>${race.race_name}</td>
                <td>${race.venue}</td>
                <td>${new Date(race.race_date).toLocaleDateString()}</td>
                <td>${race.distance}m</td>
                <td>${race.grade}</td>
                <td><span class="badge bg-secondary">Not Predicted</span></td>
                <td><button class="btn btn-sm btn-primary predict-btn" data-race-id="${race.race_id}" ${raceFilenamAttr}>Predict</button></td>
            `;
            elements.racesTableBody.appendChild(row);
        });

        // Enhanced prediction buttons are handled by PredictionButtonManager
        // No need to add individual event listeners here

        renderPagination(filteredRaces.length);
    }

    // Toggle view mode between regular and upcoming races
    function toggleViewMode() {
        state.viewMode = state.viewMode === 'regular' ? 'upcoming' : 'regular';
        
        // Update UI to reflect the new view mode
        if (elements.viewModeLabel) {
            elements.viewModeLabel.textContent = state.viewMode === 'upcoming' ? 'Upcoming Races' : 'Regular Races';
        }
        
        if (elements.viewToggleButton) {
            elements.viewToggleButton.innerHTML = `
                <i class="fas fa-${state.viewMode === 'upcoming' ? 'history' : 'calendar-plus'}"></i> 
                Switch to ${state.viewMode === 'upcoming' ? 'Regular' : 'Upcoming'}
            `;
        }
        
        // Reset pagination and reload races
        state.currentPage = 1;
        loadRaces().then(() => {
            renderRaces();
        });
    }

    // Setup event listeners
    function setupEventListeners() {
        elements.searchButton.addEventListener('click', () => {
            state.searchQuery = elements.searchBox.value;
            renderRaces();
        });

        elements.selectAllCheckbox.addEventListener('change', (e) => {
            const checkboxes = document.querySelectorAll('.race-checkbox');
            checkboxes.forEach(checkbox => checkbox.checked = e.target.checked);
        });

        elements.runSelectedButton.addEventListener('click', () => {
            const selectedRaces = Array.from(document.querySelectorAll('.race-checkbox:checked'))
                                        .map(cb => ({
                                            raceId: cb.dataset.raceId,
                                            raceFilename: cb.dataset.raceFilename
                                        }));
            if (selectedRaces.length > 0) {
                runPredictions(selectedRaces);
            }
        });
        
        elements.runAllUpcomingButton.addEventListener('click', () => {
            runAllUpcomingPredictions();
        });
        
        // Add view toggle event listener
        if (elements.viewToggleButton) {
            elements.viewToggleButton.addEventListener('click', toggleViewMode);
        }
    }

    // Filter and sort races based on state
    function filterAndSortRaces() {
        let filtered = state.races;

        if (state.searchQuery) {
            const query = state.searchQuery.toLowerCase();
            filtered = filtered.filter(race => 
                race.race_name.toLowerCase().includes(query) || 
                race.venue.toLowerCase().includes(query)
            );
        }

        const [sortKey, sortDir] = state.sortOrder.split('|');
        filtered.sort((a, b) => {
            if (a[sortKey] < b[sortKey]) return sortDir === 'asc' ? -1 : 1;
            if (a[sortKey] > b[sortKey]) return sortDir === 'asc' ? 1 : -1;
            return 0;
        });

        return filtered;
    }

    // Paginate races
    function paginateRaces(races) {
        const start = (state.currentPage - 1) * state.racesPerPage;
        const end = start + state.racesPerPage;
        return races.slice(start, end);
    }

    // Render pagination controls
    function renderPagination(totalRaces) {
        const totalPages = Math.ceil(totalRaces / state.racesPerPage);
        elements.paginationControls.innerHTML = '';

        for (let i = 1; i <= totalPages; i++) {
            const button = document.createElement('button');
            button.className = `btn btn-sm ${i === state.currentPage ? 'btn-primary' : 'btn-outline-primary'}`;
            button.textContent = i;
            button.addEventListener('click', () => {
                state.currentPage = i;
                renderRaces();
            });
            elements.paginationControls.appendChild(button);
        }
    }
    
    async function runAllUpcomingPredictions() {
        if (state.isLoading) {
            showToast('Another operation is in progress. Please wait.', 'warning');
            return;
        }
        
        const button = elements.runAllUpcomingButton;
        const originalText = button.innerHTML;
        
        try {
            state.isLoading = true;
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running All Predictions...';
            
            const response = await fetchWithErrorHandling('/api/predict_all_upcoming_races_enhanced', { 
                method: 'POST' 
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                showToast(errorText || 'Failed to run all upcoming predictions', 'danger');
                return;
            }
            
            const data = await response.json();

            if (data.success) {
                // Handle errors from the response
                if (data.errors && data.errors.length > 0) {
                    data.errors.forEach(error => {
                        showToast(error, 'warning');
                    });
                }
                
                showToast(`Successfully processed ${data.total_races || 0} races. Success: ${data.success_count || 0}, Failed: ${data.failed_count || 0}`, 'success');
                
                // Transform the batch response to individual results for display
                const individualResults = data.predictions || [];
                displayPredictionResults(individualResults);
            } else {
                // Handle errors in the response
                if (data.errors && data.errors.length > 0) {
                    data.errors.forEach(error => {
                        showToast(error, 'danger');
                    });
                }
                throw new Error(data.message || 'Unknown error occurred');
            }
        } catch (error) {
            console.error('Error running all upcoming predictions:', error);
            showToast(`Error running all upcoming predictions: ${error.message}`, 'danger');
        } finally {
            state.isLoading = false;
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }

    // Run prediction for a single race with enhanced error handling
    async function runSinglePrediction(raceId, raceFilename) {
        if (state.isLoading) {
            showToast('Another operation is in progress. Please wait.', 'warning');
            return;
        }
        
        try {
            state.isLoading = true;
            
            // Show prediction results container
            elements.predictionResultsContainer.style.display = 'block';
            elements.predictionResultsBody.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Processing single race prediction...</p>
                </div>`;
            
            // Prepare request body - send race_filename if available (for upcoming races), otherwise race_id
            const requestBody = {};
            if (raceFilename) {
                requestBody.race_filename = raceFilename;
            } else {
                requestBody.race_id = raceId;
            }
            
            const response = await fetchWithErrorHandling('/api/predict_single_race_enhanced', {
                method: 'POST',
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                showToast('Prediction completed successfully!', 'success');
                displayPredictionResults([data]);
            } else {
                showToast(`Prediction failed: ${data.message || 'Unknown error'}`, 'danger');
                displayPredictionResults([data]);
            }
            
        } catch (error) {
            console.error('Error in runSinglePrediction:', error);
            showToast(`Error running prediction: ${error.message}`, 'danger');
            elements.predictionResultsBody.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${error.message}
                </div>`;
        } finally {
            state.isLoading = false;
        }
    }

    // Run predictions for selected races with improved error handling
    async function runPredictions(races) {
        if (!Array.isArray(races) || races.length === 0) {
            showToast('No races selected for prediction', 'warning');
            return;
        }
        
        if (state.isLoading) {
            showToast('Another operation is in progress. Please wait.', 'warning');
            return;
        }
        
        const button = elements.runSelectedButton;
        const originalText = button.innerHTML;
        
        try {
            state.isLoading = true;
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Predictions...';
            
            elements.predictionResultsContainer.style.display = 'block';
            elements.predictionResultsBody.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Processing ${races.length} race(s)...</p>
                </div>`;

            const results = [];
            let successCount = 0;
            
            for (let i = 0; i < races.length; i++) {
                const race = races[i];
                const raceId = race.raceId;
                const raceFilename = race.raceFilename;
                
                // Update progress
                const displayId = raceFilename || raceId;
                elements.predictionResultsBody.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2">Processing race ${i + 1} of ${races.length} (${displayId})...</p>
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${((i) / races.length) * 100}%"
                                 aria-valuenow="${i}" aria-valuemin="0" aria-valuemax="${races.length}"></div>
                        </div>
                    </div>`;
                
                try {
                    // Prepare request body - send race_filename if available (for upcoming races), otherwise race_id
                    const requestBody = {};
                    if (raceFilename) {
                        requestBody.race_filename = raceFilename;
                    } else {
                        requestBody.race_id = raceId;
                    }
                    
                    const response = await fetchWithErrorHandling('/api/predict_single_race_enhanced', {
                        method: 'POST',
                        body: JSON.stringify(requestBody)
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText || `HTTP ${response.status}`);
                    }
                    
                    const data = await response.json();
                    results.push(data);
                    
                    if (data.success) {
                        successCount++;
                    }
                } catch (error) {
                    console.error(`Error predicting race ${displayId}:`, error);
                    results.push({ 
                        success: false, 
                        race_id: raceId,
                        race_filename: raceFilename,
                        message: error.message,
                        error_type: 'network_error'
                    });
                }
            }

            showToast(`Completed predictions: ${successCount}/${races.length} successful`, 
                      successCount === races.length ? 'success' : 'warning');
            displayPredictionResults(results);
            
        } catch (error) {
            console.error('Error in runPredictions:', error);
            showToast(`Error running predictions: ${error.message}`, 'danger');
            elements.predictionResultsBody.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${error.message}
                </div>`;
        } finally {
            state.isLoading = false;
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }

    // Display prediction results with enhanced formatting
    function displayPredictionResults(results) {
        if (!Array.isArray(results) || results.length === 0) {
            elements.predictionResultsBody.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> No prediction results to display.
                </div>`;
            return;
        }
        
        // Store results globally for details expansion
        window.currentPredictionResults = results;
        
        elements.predictionResultsBody.innerHTML = '';
        
        results.forEach((result, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = `alert ${result.success ? 'alert-success' : 'alert-danger'} mb-3`;
            
            if (result.success && result.predictions && result.predictions.length > 0) {
                const predictions = result.predictions;
                const topPick = predictions[0]; // First prediction is the top pick
                
                if (topPick) {
                    const winProb = topPick.final_score || topPick.win_probability || topPick.confidence || 0;
                    const dogName = topPick.dog_name || topPick.name || 'Unknown';
                    const raceInfo = result.race_filename || result.race_id || `Race ${index + 1}`;
                    
                    resultDiv.innerHTML = `
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
                                    ${predictions.length} dogs analyzed
                                    ${result.predictor_used ? ` | Predictor: ${result.predictor_used}` : ''}
                                </small>
                            </div>
                            <button class="btn btn-sm btn-outline-primary" 
                                    onclick="toggleDetails(this, ${index})" 
                                    data-expanded="false">
                                <i class="fas fa-chevron-down"></i> Details
                            </button>
                        </div>
                        <div class="prediction-details" id="details-${index}" style="display: none; margin-top: 15px;">
                            <!-- Details will be populated when expanded -->
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <h6 class="alert-heading">
                            <i class="fas fa-check-circle"></i> ${result.race_id || `Race ${index + 1}`}
                        </h6>
                        <p class="mb-0">Prediction completed but no predictions available.</p>
                    `;
                }
            } else {
                const errorMessage = result.message || result.error || 'Unknown error occurred';
                const raceInfo = result.race_id || `Race ${index + 1}`;
                
                resultDiv.innerHTML = `
                    <h6 class="alert-heading">
                        <i class="fas fa-exclamation-triangle"></i> ${raceInfo}
                    </h6>
                    <p class="mb-0"><strong>Error:</strong> ${errorMessage}</p>
                    ${result.error_type === 'network_error' ? 
                        '<small class="text-muted">This may be a temporary network issue. Please try again.</small>' : ''}
                `;
            }
            
            elements.predictionResultsBody.appendChild(resultDiv);
        });
        
        // Add summary at the bottom
        const successCount = results.filter(r => r.success).length;
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'alert alert-info mt-3';
        summaryDiv.innerHTML = `
            <h6><i class="fas fa-info-circle"></i> Summary</h6>
            <p class="mb-0">
                Processed ${results.length} race(s): 
                <span class="badge bg-success">${successCount} successful</span>
                <span class="badge bg-danger">${results.length - successCount} failed</span>
            </p>
        `;
        elements.predictionResultsBody.appendChild(summaryDiv);
    }
    
    // Toggle prediction details
    window.toggleDetails = function(button, index) {
        const detailsDiv = document.getElementById(`details-${index}`);
        const icon = button.querySelector('i');
        const isExpanded = button.getAttribute('data-expanded') === 'true';
        
        if (isExpanded) {
            detailsDiv.style.display = 'none';
            icon.className = 'fas fa-chevron-down';
            button.innerHTML = '<i class="fas fa-chevron-down"></i> Details';
            button.setAttribute('data-expanded', 'false');
        } else {
            detailsDiv.style.display = 'block';
            icon.className = 'fas fa-chevron-up';
            button.innerHTML = '<i class="fas fa-chevron-up"></i> Hide';
            button.setAttribute('data-expanded', 'true');
            
            // Load detailed information if not already loaded
            if (detailsDiv.innerHTML.trim() === '') {
                detailsDiv.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading details...</div>';
                
                // Get the result data from the current results array
                const resultIndex = parseInt(index);
                const currentResults = window.currentPredictionResults || [];
                const result = currentResults[resultIndex];
                
                setTimeout(() => {
                    if (result && result.predictions && result.predictions.length > 0) {
                        let detailsHTML = '<div class="row">';
                        
                        result.predictions.forEach((prediction, idx) => {
                            const winProb = prediction.final_score || prediction.win_probability || prediction.confidence || 0;
                            const dogName = prediction.dog_name || prediction.name || 'Unknown';
                            const boxNumber = prediction.box_number || prediction.box || 'N/A';
                            
                            detailsHTML += `
                                <div class="col-md-6 mb-2">
                                    <div class="card card-sm">
                                        <div class="card-body p-2">
                                            <h6 class="card-title mb-1">${idx + 1}. ${dogName}</h6>
                                            <p class="card-text mb-1">
                                                <small>Box: ${boxNumber} | Confidence: ${(winProb * 100).toFixed(1)}%</small>
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                        
                        detailsHTML += '</div>';
                        
                        if (result.message) {
                            detailsHTML += `<div class="mt-2"><small class="text-muted"><i class="fas fa-info-circle"></i> ${result.message}</small></div>`;
                        }
                        
                        detailsDiv.innerHTML = detailsHTML;
                    } else {
                        detailsDiv.innerHTML = '<p class="text-muted">No detailed prediction data available.</p>';
                    }
                }, 500);
            }
        }
    };

    // Utility to show alerts (legacy function for compatibility)
    function showAlert(message, type = 'info') {
        showToast(message, type);
    }

    init();
});

document.addEventListener('DOMContentLoaded', function() {
    const state = {
        races: [],
        venues: [],
        grades: [],
        filters: {
            sortBy: 'race_date',
            order: 'desc',
            page: 1,
            perPage: 12,
            searchQuery: '',
            status: 'all',
            venue: 'all',
            grade: 'all',
            minDistance: '',
            maxDistance: '',
            minConfidence: 0
        },
        pagination: {},
        isLoading: false,
        view: 'grid' // 'grid' or 'list'
    };

    const elements = {
        racesContainer: document.getElementById('racesContainer'),
        loadingSpinner: document.getElementById('loadingSpinner'),
        noResultsMessage: document.getElementById('noResultsMessage'),
        searchInput: document.getElementById('searchInput'),
        sortSelect: document.getElementById('sortSelect'),
        statusFilter: document.getElementById('statusFilter'),
        venueFilter: document.getElementById('venueFilter'),
        clearSearch: document.getElementById('clearSearch'),
        toggleView: document.getElementById('toggleView'),
        viewIcon: document.getElementById('viewIcon'),
        pagination: document.getElementById('pagination'),
        paginationNav: document.getElementById('paginationNav')
    };

    function init() {
        setupEventListeners();
        loadVenuesAndGrades();
        fetchRaces();
    }

    // Debounce utility function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Fetch races function (placeholder - implement based on your API)
    async function fetchRaces() {
        // This function should be implemented to fetch races from your API
        // For now, it's a placeholder that does nothing
        console.log('fetchRaces called - implement based on your API requirements');
    }

    function setupEventListeners() {
        elements.searchInput.addEventListener('input', debounce(handleSearch, 300));
        elements.clearSearch.addEventListener('click', clearSearch);
        elements.sortSelect.addEventListener('change', handleFilterChange);
        elements.statusFilter.addEventListener('change', handleFilterChange);
        elements.venueFilter.addEventListener('change', handleFilterChange);
        elements.toggleView.addEventListener('click', toggleView);
    }

    async function loadVenuesAndGrades() {
        // In a real application, you would fetch these from an API endpoint
        state.venues = ['The Meadows', 'Sandown Park', 'Wentworth Park', 'Albion Park'];
        state.grades = ['Group 1', 'Group 2', 'Group 3', 'Maiden', 'Novice'];
        
        populateSelect(elements.venueFilter, state.venues, 'All Venues');
    }

    function populateSelect(selectElement, options, defaultOption) {
        selectElement.innerHTML = `<option value="all">${defaultOption}</option>`;
        options.forEach(option => {
            selectElement.innerHTML += `<option value="${option}">${option}</option>`;
        });
    }

    function handleSearch(event) {
        state.filters.searchQuery = event.target.value;
        state.filters.page = 1;
        fetchRaces();
    }

    function clearSearch() {
        elements.searchInput.value = '';
        state.filters.searchQuery = '';
        state.filters.page = 1;
        fetchRaces();
    }

    function handleFilterChange() {
        state.filters.sortBy = elements.sortSelect.value.split('|')[0];
        state.filters.order = elements.sortSelect.value.split('|')[1];
        state.filters.status = elements.statusFilter.value;
        state.filters.venue = elements.venueFilter.value;
        state.filters.page = 1;
        fetchRaces();
    }

    function toggleView() {
        state.view = state.view === 'grid' ? 'list' : 'grid';
        elements.viewIcon.className = state.view === 'grid' ? 'fas fa-th-large' : 'fas fa-list';
        elements.racesContainer.className = `races-container ${state.view}`;
    }

    init();
});

