// Interactive Races Page JavaScript

document.addEventListener('DOMContentLoaded', () => {
    const state = {
        races: [],
        currentPage: 1,
        racesPerPage: 10,
        searchQuery: '',
        sortOrder: 'race_date|desc',
        filters: {},
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
        await loadRaces();
        setupEventListeners();
        renderRaces();
    }

    // Load races from API
    async function loadRaces() {
        try {
            const response = await fetch('/api/races/paginated');
            const data = await response.json();
            if (data.success) {
                state.races = data.races;
            } else {
                showAlert('Failed to load races.', 'danger');
            }
        } catch (error) {
            showAlert('Error loading races: ' + error.message, 'danger');
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
        try {
            const response = await fetch('/api/predict_all_upcoming_races_enhanced', { method: 'POST' });
            const data = await response.json();

            if (data.success) {
                showAlert(`Successfully processed ${data.total_races} races.`, 'success');
                displayPredictionResults([data]);
            } else {
                showAlert(data.message || 'Failed to run all upcoming predictions.', 'danger');
            }
        } catch (error) {
            showAlert('Error running all upcoming predictions: ' + error.message, 'danger');
        }
    }

    // Run predictions for selected races
    async function runPredictions(raceIds) {
        elements.predictionResultsContainer.style.display = 'block';
        elements.predictionResultsBody.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';

        const results = [];
        for (const raceId of raceIds) {
            try {
                const response = await fetch(`/api/predict_single_race_enhanced`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ race_id: raceId })
                });
                const data = await response.json();
                results.push(data);
            } catch (error) {
                results.push({ success: false, race_id: raceId, message: error.message });
            }
        }

        displayPredictionResults(results);
    }

    // Display prediction results
    function displayPredictionResults(results) {
        elements.predictionResultsBody.innerHTML = '';
        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = `alert ${result.success ? 'alert-success' : 'alert-danger'}`;
            if (result.success) {
                const topPick = result.prediction.top_pick;
                resultDiv.innerHTML = `<strong>Race:</strong> ${result.race_id} - <strong>Top Pick:</strong> ${topPick.dog_name} (Win Probability: ${(topPick.win_probability * 100).toFixed(2)}%)`;
            } else {
                resultDiv.innerHTML = `<strong>Race:</strong> ${result.race_id} - <strong>Error:</strong> ${result.message}`;
            }
            elements.predictionResultsBody.appendChild(resultDiv);
        });
    }

    // Utility to show alerts
    function showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.querySelector('.container-fluid').insertAdjacentElement('afterbegin', alertDiv);
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

