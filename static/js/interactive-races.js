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
            const response = await fetchWithErrorHandling('/api/races/paginated');
            const data = await response.json();
            
            if (data.success) {
                state.races = Array.isArray(data.races) ? data.races : [];
                showToast(`Successfully loaded ${state.races.length} races`, 'success');
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
            row.innerHTML = `
                <td><input type="checkbox" class="race-checkbox" data-race-id="${race.race_id}"></td>
                <td>${race.race_name}</td>
                <td>${race.venue}</td>
                <td>${new Date(race.race_date).toLocaleDateString()}</td>
                <td>${race.distance}m</td>
                <td>${race.grade}</td>
                <td><span class="badge bg-secondary">Not Predicted</span></td>
                <td><button class="btn btn-sm btn-primary predict-btn" data-race-id="${race.race_id}">Predict</button></td>
            `;
            elements.racesTableBody.appendChild(row);
        });

        renderPagination(filteredRaces.length);
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
            const selectedIds = Array.from(document.querySelectorAll('.race-checkbox:checked'))
                                     .map(cb => cb.dataset.raceId);
            if (selectedIds.length > 0) {
                runPredictions(selectedIds);
            }
        });
        
        elements.runAllUpcomingButton.addEventListener('click', () => {
            runAllUpcomingPredictions();
        });
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
                showToast(`Successfully processed ${data.total_races || 0} races.`, 'success');
                displayPredictionResults([data]);
            } else {
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

    // Run predictions for selected races with improved error handling
    async function runPredictions(raceIds) {
        if (!Array.isArray(raceIds) || raceIds.length === 0) {
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
                    <p class="mt-2">Processing ${raceIds.length} race(s)...</p>
                </div>`;

            const results = [];
            let successCount = 0;
            
            for (let i = 0; i < raceIds.length; i++) {
                const raceId = raceIds[i];
                
                // Update progress
                elements.predictionResultsBody.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2">Processing race ${i + 1} of ${raceIds.length} (Race ID: ${raceId})...</p>
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${((i) / raceIds.length) * 100}%"
                                 aria-valuenow="${i}" aria-valuemin="0" aria-valuemax="${raceIds.length}"></div>
                        </div>
                    </div>`;
                
                try {
                    const response = await fetchWithErrorHandling('/api/predict_single_race_enhanced', {
                        method: 'POST',
                        body: JSON.stringify({ race_id: raceId })
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
                    console.error(`Error predicting race ${raceId}:`, error);
                    results.push({ 
                        success: false, 
                        race_id: raceId, 
                        message: error.message,
                        error_type: 'network_error'
                    });
                }
            }

            showToast(`Completed predictions: ${successCount}/${raceIds.length} successful`, 
                      successCount === raceIds.length ? 'success' : 'warning');
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
        
        elements.predictionResultsBody.innerHTML = '';
        
        results.forEach((result, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = `alert ${result.success ? 'alert-success' : 'alert-danger'} mb-3`;
            
            if (result.success && result.prediction) {
                const prediction = result.prediction;
                const topPick = prediction.top_pick || prediction.predictions?.[0];
                
                if (topPick) {
                    const winProb = topPick.final_score || topPick.win_probability || topPick.confidence || 0;
                    const dogName = topPick.dog_name || topPick.name || 'Unknown';
                    const raceInfo = result.race_name || result.race_id || `Race ${index + 1}`;
                    
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
                                ${prediction.betting_suggestions ? 
                                    `<small class="text-muted">
                                        <i class="fas fa-lightbulb"></i> 
                                        ${prediction.betting_suggestions.length} betting suggestions available
                                    </small>` : ''}
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
                        <p class="mb-0">Prediction completed but no top pick available.</p>
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
                // Here you could load more detailed prediction information
                setTimeout(() => {
                    detailsDiv.innerHTML = '<p class="text-muted">Detailed prediction analysis would be displayed here.</p>';
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

