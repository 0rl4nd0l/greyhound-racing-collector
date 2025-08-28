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

// Prediction ordering helpers
window.predOrderingMode = window.predOrderingMode || 'win_prob';
function predictionScoreWinProb(p) {
    return Number(p.win_prob || p.normalized_win_probability || p.win_probability || p.final_score || p.prediction_score || p.confidence || 0);
}
function sortPreds(list, mode) {
    const arr = Array.isArray(list) ? [...list] : [];
    if ((mode || window.predOrderingMode) === 'predicted_rank') {
        return arr.sort((a, b) => {
            const ra = Number(a.predicted_rank ?? Number.POSITIVE_INFINITY);
            const rb = Number(b.predicted_rank ?? Number.POSITIVE_INFINITY);
            return ra - rb;
        });
    }
    return arr.sort((a, b) => predictionScoreWinProb(b) - predictionScoreWinProb(a));
}
function getTopPick(list) {
    const s = sortPreds(list, 'win_prob');
    return s.length ? s[0] : null;
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
        viewMode: 'upcoming', // 'regular' or 'upcoming'
        predictedList: [],
        predictedVenues: [],
        predictedFilters: {
            venue: 'all',
            startDate: '',
            endDate: '',
            search: ''
        }
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

    // Fallback bindings for differing template IDs/structure
    if (!elements.racesTableBody) {
        elements.racesTableBody = document.querySelector('#upcoming-races-table tbody') ||
                                   document.querySelector('tbody#upcoming-races-body') ||
                                   document.querySelector('.upcoming-races tbody') ||
                                   document.querySelector('main tbody') ||
                                   document.querySelector('tbody');
        if (elements.racesTableBody) {
            console.log('Using fallback racesTableBody selector');
        }
    }
    if (!elements.selectAllCheckbox) {
        elements.selectAllCheckbox = document.querySelector('#upcoming-races-table thead input[type="checkbox"], thead input[type="checkbox"]');
        if (elements.selectAllCheckbox) {
            console.log('Using fallback selectAllCheckbox selector');
        }
    }
    if (!elements.paginationControls) {
        elements.paginationControls = document.getElementById('upcoming-pagination') ||
                                      document.querySelector('.pagination-controls') ||
                                      document.createElement('div');
        // If we had to create a div, attach it to the DOM near the table so users can navigate pages
        try {
            if (elements.paginationControls && !document.body.contains(elements.paginationControls)) {
                if (!elements.paginationControls.id) {
                    elements.paginationControls.id = 'pagination-controls';
                }
                elements.paginationControls.classList.add('pagination-controls');
                const tableEl = elements.racesTableBody ? elements.racesTableBody.closest('table') : null;
                const container = (tableEl && tableEl.parentElement) || (elements.racesTableBody && elements.racesTableBody.parentElement) || document.querySelector('main') || document.body;
                container.appendChild(elements.paginationControls);
            }
        } catch (e) {
            console.warn('Could not attach pagination controls to DOM:', e);
        }
    }

    // Check for missing elements and log them (non-fatal except for table body)
    const missingElements = [];
    Object.keys(elements).forEach(key => {
        if (!elements[key]) {
            missingElements.push(key);
            console.warn(`Missing DOM element: ${key}`);
        }
    });

    if (!elements.racesTableBody) {
        console.error('Interactive Races: races table body not found – cannot render races');
        showToast('Page markup missing races table body. Please refresh or report this issue.', 'danger');
        return;
    }

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
            if (state.viewMode === 'upcoming') {
                // 1) Fetch predicted status list (filenames for accurate filtering)
                let predictedFilenameSet = new Set();
                let predictedNameSet = new Set();
                let predictedList = [];
                try {
                    const rs = await fetchWithErrorHandling('/api/race_files_status');
                    const rd = await rs.json();
                    const preds = Array.isArray(rd.predicted_races) ? rd.predicted_races : [];
                    predictedList = preds;
                    predictedFilenameSet = new Set(preds.map(p => (p.filename || '').trim()).filter(Boolean));
                    predictedNameSet = new Set(preds.map(p => (p.race_name || '').trim()).filter(Boolean));
                } catch (e) {
                    console.warn('Could not load predicted status:', e);
                }

                // 2) Stream upcoming races (live preferred; server falls back to CSV)
                closeUpcomingStream();
                const { races: streamed, total } = await loadUpcomingViaStream(predictedFilenameSet, predictedNameSet);

                // 3) Persist streamed unpredicted races and predicted list for panel
                state.races = Array.isArray(streamed) ? streamed : [];
                state.predictedList = predictedList;

                showToast(`Loaded ${total} upcoming; ${state.races.length} unpredicted shown. Predicted moved to Re-Predict panel.`, 'info');

                // 4) Initialize and render predicted panel with filters
                try {
                    setupPredictedFilters();
                    renderPredictedPanel();
                } catch (e) {
                    console.warn('Failed to render predicted panel:', e);
                }

            } else {
                // Regular historical/paginated view as before
                const endpoint = '/api/races/paginated';
                const response = await fetchWithErrorHandling(endpoint);
                const data = await response.json();
                if (data.success) {
                    state.races = Array.isArray(data.races) ? data.races : Object.values(data.races || {});
                    showToast(`Successfully loaded ${state.races.length} regular races`, 'success');
                } else {
                    throw new Error(data.message || 'Failed to load races');
                }
            }
        } catch (error) {
            console.error('Failed to load races:', error);
            showToast(`Error loading races: ${error.message}`, 'danger');
            state.races = [];
            // Hide predicted panel on failure
            try {
                const panel = document.getElementById('predicted-repredict-container');
                if (panel) panel.style.display = 'none';
            } catch {}
        } finally {
            state.isLoading = false;
        }
    }

    // Predicted panel: setup filter options and listeners
    function setupPredictedFilters() {
        try {
            // Build venue options from predictedList
            const venues = Array.from(new Set((state.predictedList || [])
                .map(p => (p.venue || '').trim())
                .filter(v => v && v.toLowerCase() !== 'unknown' && v.toLowerCase() !== 'n/a')
            )).sort((a, b) => a.localeCompare(b));
            state.predictedVenues = venues;

            const venueSelect = document.getElementById('predicted-filter-venue');
            if (venueSelect) {
                // Preserve current selection if possible
                const current = state.predictedFilters.venue || 'all';
                venueSelect.innerHTML = '<option value="all">All Venues</option>' +
                    venues.map(v => `<option value="${v}">${v}</option>`).join('');
                venueSelect.value = current;
            }

            // Attach listeners once
            attachPredictedFilterListeners();
        } catch (e) {
            console.warn('setupPredictedFilters failed', e);
        }
    }

    function attachPredictedFilterListeners() {
        const venueSelect = document.getElementById('predicted-filter-venue');
        const startInput = document.getElementById('predicted-filter-start');
        const endInput = document.getElementById('predicted-filter-end');
        const searchInput = document.getElementById('predicted-filter-search');
        const clearBtn = document.getElementById('predicted-filter-clear');

        if (venueSelect && !venueSelect._bound) {
            venueSelect.addEventListener('change', () => {
                state.predictedFilters.venue = venueSelect.value || 'all';
                state.currentPage = 1;
                renderPredictedPanel();
            });
            venueSelect._bound = true;
        }
        if (startInput && !startInput._bound) {
            startInput.addEventListener('change', () => {
                state.predictedFilters.startDate = startInput.value || '';
                state.currentPage = 1;
                renderPredictedPanel();
            });
            startInput._bound = true;
        }
        if (endInput && !endInput._bound) {
            endInput.addEventListener('change', () => {
                state.predictedFilters.endDate = endInput.value || '';
                state.currentPage = 1;
                renderPredictedPanel();
            });
            endInput._bound = true;
        }
        if (searchInput && !searchInput._bound) {
            const debounced = debounce(() => {
                state.predictedFilters.search = searchInput.value || '';
                state.currentPage = 1;
                renderPredictedPanel();
            }, 250);
            searchInput.addEventListener('input', debounced);
            searchInput._bound = true;
        }
        if (clearBtn && !clearBtn._bound) {
            clearBtn.addEventListener('click', () => {
                state.predictedFilters = { venue: 'all', startDate: '', endDate: '', search: '' };
                if (venueSelect) venueSelect.value = 'all';
                if (startInput) startInput.value = '';
                if (endInput) endInput.value = '';
                if (searchInput) searchInput.value = '';
                renderPredictedPanel();
            });
            clearBtn._bound = true;
        }
    }

    function parseDateSafe(s) {
        if (!s) return null;
        try {
            // Accept YYYY-MM-DD or ISO
            const d = new Date(s);
            return isNaN(d.getTime()) ? null : d;
        } catch { return null; }
    }

    function getFilteredPredicted() {
        const list = Array.isArray(state.predictedList) ? state.predictedList.slice() : [];
        const { venue, startDate, endDate, search } = state.predictedFilters;
        const sDate = parseDateSafe(startDate);
        const eDate = parseDateSafe(endDate);
        const q = (search || '').toLowerCase();

        let filtered = list.filter(item => {
            // Venue
            if (venue && venue !== 'all') {
                if ((item.venue || '').trim() !== venue) return false;
            }
            // Date range (based on race_date)
            if (sDate || eDate) {
                const rd = parseDateSafe(item.race_date);
                if (!rd) return false;
                if (sDate && rd < sDate) return false;
                if (eDate) {
                    // Include end date full-day
                    const endDay = new Date(eDate.getTime());
                    endDay.setHours(23, 59, 59, 999);
                    if (rd > endDay) return false;
                }
            }
            // Search in race name or venue
            if (q) {
                const hay = `${item.race_name || ''} ${item.venue || ''}`.toLowerCase();
                if (!hay.includes(q)) return false;
            }
            return true;
        });

        // Sort by prediction timestamp desc then file_mtime desc
        filtered.sort((a, b) => {
            const ta = parseDateSafe(a.prediction_timestamp)?.getTime() || Number(a.file_mtime) || 0;
            const tb = parseDateSafe(b.prediction_timestamp)?.getTime() || Number(b.file_mtime) || 0;
            return tb - ta;
        });

        return filtered;
    }

    function renderPredictedPanel() {
        const panel = document.getElementById('predicted-repredict-container');
        const tbody = document.getElementById('predicted-repredict-body');
        const empty = document.getElementById('predicted-repredict-empty');
        const summary = document.getElementById('predicted-filter-summary');
        if (!panel || !tbody) return;

        const filtered = getFilteredPredicted();
        const total = Array.isArray(state.predictedList) ? state.predictedList.length : 0;
        const limit = 20; // show top 20 after filtering
        const toShow = filtered.slice(0, limit);

        tbody.innerHTML = '';
        if (toShow.length === 0) {
            if (empty) empty.style.display = 'block';
            panel.style.display = total > 0 ? 'block' : 'none';
        } else {
            if (empty) empty.style.display = 'none';
            toShow.forEach(item => {
                const filename = (item.filename || item.race_filename || '').trim();
                const displayName = item.race_name || filename || '';
                const venue = item.venue || '';
                const date = item.race_date || '';
                const distance = item.distance || '';
                const grade = item.grade || '';
                const ts = item.prediction_timestamp || '';
                const ridAttr = item.race_id ? ` data-race-id="${item.race_id}"` : '';
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${displayName}</td>
                    <td>${venue}</td>
                    <td>${date}</td>
                    <td>${distance}</td>
                    <td>${grade}</td>
                    <td><small>${ts || ''}</small></td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary predict-btn" data-race-filename="${filename}"${ridAttr}>
                            <i class="fas fa-redo"></i> Re-Predict
                        </button>
                    </td>
                `;
                tbody.appendChild(row);
            });
            panel.style.display = 'block';
        }

        if (summary) {
            const showing = toShow.length;
            summary.textContent = `Showing ${showing} of ${filtered.length} (total predicted: ${total})`;
        }
    }

    // Render races in the table
    function renderRaces() {
        if (!elements.racesTableBody) {
            console.error('racesTableBody element not found');
            return;
        }
        
        const filteredRaces = filterAndSortRaces();
        const paginatedRaces = paginateRaces(filteredRaces);

        elements.racesTableBody.innerHTML = '';
        if (paginatedRaces.length === 0) {
            // Determine column span from header if possible
            const table = elements.racesTableBody.closest('table');
            const headerCols = table && table.querySelectorAll('thead th') ? table.querySelectorAll('thead th').length : 8;
            elements.racesTableBody.innerHTML = `<tr><td colspan="${headerCols}" class="text-center">No races found.</td></tr>`;
            return;
        }

        paginatedRaces.forEach(race => {
            const row = document.createElement('tr');
            
            // Add race_filename data attribute for upcoming races
            const raceFilenamAttr = race.filename ? `data-race-filename="${race.filename}"` : '';

            // Melbourne-aware date + time display
            const dt = getDateAndTimeDisplay(race);
            const dateCell = dt.dateText ? `${dt.dateText}${dt.timeText ? ' ' + dt.timeText : ' (TBD)'}` : '';
            
            // Race number (accept multiple field names)
            const raceNum = (race.race_number ?? race.number ?? race.race_no ?? race.race ?? '').toString().trim();
            const raceNumLabel = raceNum ? `R${raceNum} — ` : '';
            
            // Build actions based on whether a local CSV is present
            const hasLocalCsv = !!(race.filename && String(race.filename).trim());
            const actionsHtml = hasLocalCsv
                ? `<button class="btn btn-sm btn-primary predict-btn" data-race-id="${race.race_id}" ${raceFilenamAttr}>Predict</button>`
                : `<button class="btn btn-sm btn-warning download-predict-btn" 
                        data-venue="${safeVenue(race.venue)}" 
                        data-race-number="${raceNum}"
                        data-race-url="${String(race.url || '')}">
                        <i class="fas fa-download"></i> Download + Predict
                   </button>`;

            row.innerHTML = `
                <td><input type="checkbox" class="race-checkbox" data-race-id="${race.race_id}" ${raceFilenamAttr}></td>
                <td>${raceNumLabel}${race.race_name || ''}</td>
                <td>${safeVenue(race.venue)}</td>
                <td>${dateCell}</td>
                <td>${safeDistance(race.distance) ? safeDistance(race.distance) + 'm' : ''}</td>
                <td>${safeGrade(race.grade)}</td>
                <td><span class="badge bg-secondary">Not Predicted</span></td>
                <td>${actionsHtml}</td>
            `;

            // Attach useful data attributes for downstream use/testing
            try {
                const tsSec = computeMelbourneTimestampSeconds(race);
                if (Number.isFinite(tsSec)) row.setAttribute('data-race-timestamp', String(tsSec));
                if (race.race_date || race.date) row.setAttribute('data-race-date', String(race.race_date || race.date));
                if (race.source) row.setAttribute('data-source', String(race.source));
            } catch {}

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
        try { closeUpcomingStream(); } catch {}
        loadRaces().then(() => {
            renderRaces();
        });
    }

    // Setup event listeners
    function setupEventListeners() {
        if (elements.searchButton) {
            elements.searchButton.addEventListener('click', () => {
                state.searchQuery = elements.searchBox ? elements.searchBox.value : '';
                renderRaces();
            });
        }

        if (elements.selectAllCheckbox) {
            elements.selectAllCheckbox.addEventListener('change', (e) => {
                const table = elements.racesTableBody ? elements.racesTableBody.closest('table') : null;
                const scope = table || document;
                const checkboxes = scope.querySelectorAll('.race-checkbox');
                checkboxes.forEach(checkbox => checkbox.checked = e.target.checked);
            });
        }

        if (elements.runSelectedButton) {
            elements.runSelectedButton.addEventListener('click', () => {
                const selectedRaces = Array.from(document.querySelectorAll('.race-checkbox:checked'))
                                            .map(cb => ({
                                                raceId: cb.dataset.raceId,
                                                raceFilename: cb.dataset.raceFilename
                                            }));
                if (selectedRaces.length > 0) {
                    runPredictions(selectedRaces);
                } else {
                    showToast('Please choose at least one race', 'warning');
                }
            });
        }
        
        if (elements.runAllUpcomingButton) {
            elements.runAllUpcomingButton.addEventListener('click', () => {
                runAllUpcomingPredictions();
            });
        }
        
        // Add view toggle event listener
        if (elements.viewToggleButton) {
            elements.viewToggleButton.addEventListener('click', toggleViewMode);
        }

        // Event delegation for Download + Predict, Download CSV, and fallback Re-Predict actions
        document.addEventListener('click', async (evt) => {
            const dlPredictBtn = evt.target.closest('.download-predict-btn');
            if (dlPredictBtn) {
                evt.preventDefault();
                const venue = dlPredictBtn.getAttribute('data-venue') || '';
                const raceNumber = dlPredictBtn.getAttribute('data-race-number') || '';
                try {
                    await downloadAndPredictRace(dlPredictBtn, { venue, raceNumber });
                } catch (e) {
                    console.warn('downloadAndPredictRace failed', e);
                }
                return;
            }
            const dlCsvBtn = evt.target.closest('.download-csv-btn');
            if (dlCsvBtn) {
                evt.preventDefault();
                const raceUrl = dlCsvBtn.getAttribute('data-race-url') || '';
                try {
                    await downloadCsvForRace(dlCsvBtn, { raceUrl });
                } catch (e) {
                    console.warn('downloadCsvForRace failed', e);
                }
                return;
            }
            // Fallback: handle Re-Predict buttons here if the global PredictionButtonManager is not present
            const rePredictBtn = evt.target.closest('.predict-btn');
            if (rePredictBtn && (typeof window.PredictionButtonManager === 'undefined')) {
                evt.preventDefault();
                try {
                    const raceFilename = rePredictBtn.getAttribute('data-race-filename') || '';
                    const raceId = rePredictBtn.getAttribute('data-race-id') || '';
                    await runSinglePrediction(raceId, raceFilename);
                } catch (e) {
                    console.warn('fallback runSinglePrediction failed', e);
                }
                return;
            }
        });
    }

    // Download only helper (optional standalone use)
    async function downloadCsvForRace(button, { raceUrl }) {
        if (!raceUrl) {
            showToast('Missing race URL for download', 'warning');
            return;
        }
        const original = button.innerHTML;
        try {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';
            const resp = await fetchWithErrorHandling('/api/download_upcoming_race', {
                method: 'POST',
                body: JSON.stringify({ race_url: raceUrl })
            });
            const data = await resp.json();
            if (data && data.success) {
                showToast(`Downloaded ${data.filename || 'race CSV'}`, 'success');
            } else {
                showToast(`Download failed: ${(data && (data.error || data.message)) || 'Unknown error'}`, 'danger');
            }
        } catch (e) {
            showToast(`Download error: ${e.message}`, 'danger');
        } finally {
            button.disabled = false;
            button.innerHTML = original;
        }
    }

    // Download + Predict helper
    async function downloadAndPredictRace(button, { venue, raceNumber }) {
        if (!venue || !String(raceNumber).trim()) {
            showToast('Missing venue or race number', 'warning');
            return;
        }
        const original = button.innerHTML;
        try {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Working...';
            const resp = await fetchWithErrorHandling('/api/download_and_predict_race', {
                method: 'POST',
                body: JSON.stringify({ venue, race_number: String(raceNumber).trim() })
            });
            const data = await resp.json();
            if (!data || data.success !== true) {
                const msg = (data && (data.message || data.error)) || 'Unknown error';
                showToast(`Download + Predict failed: ${msg}`, 'danger');
                return;
            }
            showToast('Download + Predict completed', data.prediction_success ? 'success' : 'warning');

            // Try to render detailed results if a prediction file was produced
            const filenameNoExt = String(data.filename || '').replace(/\.csv$/i, '');
            if (filenameNoExt) {
                try {
                    const detailResp = await fetchWithErrorHandling(`/api/prediction_detail/${encodeURIComponent(filenameNoExt)}`);
                    const detailData = await detailResp.json();
                    if (detailData && detailData.success && detailData.prediction) {
                        // Reuse existing display function
                        displayPredictionResults([{ success: true, prediction: detailData.prediction }]);
                    } else {
                        // Fallback to summary-only display
                        displayPredictionResults([{ success: true, predictions: [], message: 'Prediction completed (no detailed JSON found yet)' }]);
                    }
                } catch (e) {
                    console.warn('prediction_detail fetch failed, showing summary', e);
                    displayPredictionResults([{ success: true, predictions: [], message: 'Prediction completed (details not available yet)' }]);
                }
            }
        } catch (e) {
            showToast(`Download + Predict error: ${e.message}`, 'danger');
        } finally {
            button.disabled = false;
            button.innerHTML = original;
        }
    }

    // Convert race time to comparable format
    function convertToComparableTime(raceTime) {
        if (!raceTime) return "TBD";
        try {
            const timeStr = String(raceTime).trim();
            if (timeStr.match(/\d{1,2}:\d{2}\s*[APap][Mm]/)) {
                // Parse 12-hour format (e.g., "6:31 PM")
                const [time, period] = timeStr.split(/\s+/);
                const [hours, minutes] = time.split(':').map(Number);
                let hour24 = hours;
                if (period.toUpperCase() === 'PM' && hours !== 12) {
                    hour24 += 12;
                } else if (period.toUpperCase() === 'AM' && hours === 12) {
                    hour24 = 0;
                }
                return hour24 * 60 + minutes;
            } else if (timeStr.match(/\d{1,2}:\d{2}/)) {
                // Parse 24-hour format (e.g., "18:31")
                const [hours, minutes] = timeStr.split(':').map(Number);
                return hours * 60 + minutes;
            }
        } catch (e) {
            console.warn('Error parsing race time:', raceTime, e);
        }
        return "TBD";
    }

    // Compute a combined start timestamp for 'next to jump' ordering
    function getRaceStartTimestamp(race) {
        try {
            // Prefer race_time, but support common alternates if present
            const timeStr = race.race_time || race.scheduled_time || race.start_time || '';
            const minutes = convertToComparableTime(timeStr);
            // Return null if no valid time
            if (minutes === 'TBD' || minutes === null || typeof minutes !== 'number' || isNaN(minutes)) {
                return null;
            }
            const d = parseDateSafe(race.race_date);
            if (!d) return null;
            const base = new Date(d.getFullYear(), d.getMonth(), d.getDate(), 0, 0, 0, 0);
            return base.getTime() + (minutes * 60 * 1000);
        } catch (e) {
            console.warn('getRaceStartTimestamp failed for race:', race, e);
            return null;
        }
    }

    // Query param helper
    function getQueryParam(name) {
        try {
            const params = new URLSearchParams(window.location.search || '');
            return params.get(name);
        } catch { return null; }
    }

    // Compute Melbourne timestamp in seconds if available, else null
    function computeMelbourneTimestampSeconds(race) {
        try {
            const v = race && race.race_timestamp_melbourne;
            const num = Number(v);
            if (Number.isFinite(num) && num > 0) return num;
            const iso = race && race.race_datetime_melbourne_iso;
            if (iso) {
                const d = new Date(iso);
                const ms = d.getTime();
                if (Number.isFinite(ms)) return Math.floor(ms / 1000);
            }
        } catch {}
        return null;
    }

    // Format Melbourne date/time for display
    function getDateAndTimeDisplay(race) {
        const tsSec = computeMelbourneTimestampSeconds(race);
        if (Number.isFinite(tsSec)) {
            const d = new Date(tsSec * 1000);
            const dateText = d.toLocaleDateString('en-AU', { timeZone: 'Australia/Melbourne' });
            const timeText = d.toLocaleTimeString('en-AU', { timeZone: 'Australia/Melbourne', hour: '2-digit', minute: '2-digit', hour12: false });
            return { dateText, timeText };
        }
        const ymd = race && (race.race_date || race.date || '');
        return { dateText: ymd || '', timeText: '' };
    }

    // Build a Melbourne-aware sort key for upcoming races
    function buildUpcomingSortKey(race) {
        const ymd = (race && (race.race_date || race.date || '')) || '';
        const tsSec = computeMelbourneTimestampSeconds(race);
        const hasTime = Number.isFinite(tsSec);
        // Sort by date asc, timed first within the day, then time asc, then venue
        return [ymd, hasTime ? 0 : 1, hasTime ? tsSec : Number.POSITIVE_INFINITY, (race && (race.venue || '')) || ''];
    }

    // Start SSE stream for upcoming races (live by default, server will fallback to CSV if needed)
    async function loadUpcomingViaStream(predictedFilenameSet, predictedNameSet) {
        return new Promise((resolve) => {
            const days = Number(getQueryParam('days')) || 2;
            const requestedSource = getQueryParam('source') || 'live';
            const strictLive = getQueryParam('strict_live') || '';
            const url = `/api/upcoming_races_stream?source=${encodeURIComponent(requestedSource)}&days=${days}${strictLive ? `&strict_live=${encodeURIComponent(strictLive)}` : ''}`;
            const es = new EventSource(url);
            const map = new Map();
            let total = 0;

            try { state._sse = es; } catch {}

            es.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data || '{}');
                    if (!data || !data.type) return;
                    if (data.type === 'status') {
                        if (data.fallback_reason) {
                            showToast(`Source: ${data.source} (requested ${data.requested_source}); fallback: ${data.fallback_reason}`, 'info');
                        }
                    } else if (data.type === 'race' && data.race) {
                        const r = data.race;
                        total = data.total_found || total;
                        const fn = (r.filename || '').trim();
                        const rn = (r.race_name || '').trim();
                        if (fn && predictedFilenameSet && predictedFilenameSet.has(fn)) {
                            return; // skip predicted by filename
                        }
                        if (!fn && rn && predictedNameSet && predictedNameSet.has(rn)) {
                            return; // optional skip by race name
                        }
                        const key = fn || r.race_id || `${r.venue || ''}_${r.race_date || r.date || ''}_${r.race_number || ''}`;
                        r._mel_ts_sec = computeMelbourneTimestampSeconds(r);
                        map.set(key, r);
                    } else if (data.type === 'completion' || data.type === 'complete') {
                        // Accept both 'completion' (old client expectation) and 'complete' (server emission)
                        try { es.close(); } catch {}
                        try { state._sse = null; } catch {}
                        const arrUnsorted = Array.from(map.values());
                        const nowSec = Math.floor(Date.now() / 1000);
                        const melYmd = new Intl.DateTimeFormat('en-CA', { timeZone: 'Australia/Melbourne', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
                        const arr = arrUnsorted.filter((r) => {
                            try {
                                const ts = Number.isFinite(r._mel_ts_sec) ? r._mel_ts_sec : computeMelbourneTimestampSeconds(r);
                                if (Number.isFinite(ts)) return ts >= nowSec;
                                const ymd = String(r.race_date || r.date || '').slice(0, 10);
                                if (!ymd) return true; // keep if date unknown
                                return ymd >= melYmd;  // keep today or future
                            } catch { return true; }
                        });
                        arr.sort((a, b) => {
                            const [d1, unk1, t1, v1] = buildUpcomingSortKey(a);
                            const [d2, unk2, t2, v2] = buildUpcomingSortKey(b);
                            if (d1 !== d2) return d1 < d2 ? -1 : 1;  // date asc
                            if (unk1 !== unk2) return unk1 - unk2;   // timed first
                            if (t1 !== t2) return t1 - t2;           // time asc
                            return String(v1).localeCompare(String(v2));
                        });
                        resolve({ races: arr, total: total || arr.length });
                    }
                } catch (e) {
                    console.warn('SSE parse error', e);
                }
            };

            es.onerror = () => {
                try { es.close(); } catch {}
                try { state._sse = null; } catch {}
                resolve({ races: [], total: 0 });
            };
        });
    }

    function closeUpcomingStream() {
        try {
            if (state && state._sse) {
                state._sse.close();
                state._sse = null;
            }
        } catch {}
    }

    // Null-safe venue helper
    const safeVenue = v => (v ?? '');

    // Null-safe grade helper  
    const safeGrade = g => (g ?? '');

    // Null-safe distance helper
    const safeDistance = d => (d ?? '');

    // Filter and sort races based on state
    function filterAndSortRaces() {
        let filtered = state.races;

        if (state.searchQuery) {
            const query = state.searchQuery.toLowerCase();
            filtered = filtered.filter(race => 
                (race.race_name || '').toLowerCase().includes(query) || 
                safeVenue(race.venue).toLowerCase().includes(query)
            );
        }

        // Upcoming view: sort by Melbourne local date asc, timed races first within the same date, time asc; unknown times last
        if (state.viewMode === 'upcoming') {
            // Filter out races that have already started (Melbourne time). If time is unknown, drop only if date is before today.
            const nowSec = Math.floor(Date.now() / 1000);
            const melYmd = new Intl.DateTimeFormat('en-CA', { timeZone: 'Australia/Melbourne', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
            filtered = filtered.filter((r) => {
                try {
                    const ts = computeMelbourneTimestampSeconds(r);
                    if (Number.isFinite(ts)) return ts >= nowSec;
                    const ymd = String(r.race_date || r.date || '').slice(0, 10);
                    if (!ymd) return true; // keep if date unknown
                    return ymd >= melYmd;  // keep today or future
                } catch { return true; }
            });
            filtered.sort((a, b) => {
                const [d1, unk1, t1, v1] = buildUpcomingSortKey(a);
                const [d2, unk2, t2, v2] = buildUpcomingSortKey(b);
                if (d1 !== d2) return d1 < d2 ? -1 : 1;  // date asc
                if (unk1 !== unk2) return unk1 - unk2;   // timed first
                if (t1 !== t2) return t1 - t2;           // time asc
                return String(v1).localeCompare(String(v2));
            });
            return filtered;
        }

        // Regular view: honor the selected sort order
        const [sortKey, sortDir] = state.sortOrder.split('|');
        
        // Enhanced sorting with null safety
        if (sortKey === 'race_time') {
            // Special handling for race time sorting with null safety
            filtered.sort((a, b) => {
                const tA = convertToComparableTime(a.race_time);
                const tB = convertToComparableTime(b.race_time);
                
                // Handle "TBD" values - put them at the end
                if (tA === "TBD" && tB === "TBD") {
                    return safeVenue(a.venue).localeCompare(safeVenue(b.venue));
                }
                if (tA === "TBD") return 1; // TBD values go to end
                if (tB === "TBD") return -1; // TBD values go to end
                
                // Both are numeric, sort normally
                if (tA !== tB) return sortDir === 'asc' ? tA - tB : tB - tA;
                // Secondary sort by venue
                return safeVenue(a.venue).localeCompare(safeVenue(b.venue));
            });
        } else {
            // Standard sorting with null safety
            filtered.sort((a, b) => {
                const valueA = a[sortKey] ?? '';
                const valueB = b[sortKey] ?? '';
                if (valueA < valueB) return sortDir === 'asc' ? -1 : 1;
                if (valueA > valueB) return sortDir === 'asc' ? 1 : -1;
                return 0;
            });
        }

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
        if (!elements.paginationControls) {
            console.error('paginationControls element not found');
            return;
        }
        
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
            if (elements.predictionResultsContainer) {
                elements.predictionResultsContainer.style.display = 'block';
            }
            if (elements.predictionResultsBody) {
                elements.predictionResultsBody.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2">Processing single race prediction...</p>
                    </div>`;
            }
            
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
            if (elements.predictionResultsBody) {
                elements.predictionResultsBody.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> ${error.message}
                    </div>`;
            }
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
            
            if (elements.predictionResultsContainer) {
                elements.predictionResultsContainer.style.display = 'block';
            }
            if (elements.predictionResultsBody) {
                elements.predictionResultsBody.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2">Processing ${races.length} race(s)...</p>
                    </div>`;
            }

            const results = [];
            let successCount = 0;
            
            for (let i = 0; i < races.length; i++) {
                const race = races[i];
                const raceId = race.raceId;
                const raceFilename = race.raceFilename;
                
                // Update progress
                const displayId = raceFilename || raceId;
                if (elements.predictionResultsBody) {
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
                }
                
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
            if (elements.predictionResultsBody) {
                elements.predictionResultsBody.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> ${error.message}
                    </div>`;
            }
        } finally {
            state.isLoading = false;
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }

    // Display prediction results with enhanced formatting
    function displayPredictionResults(results) {
        if (!elements.predictionResultsBody) {
            console.error('predictionResultsBody element not found');
            return;
        }
        
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

        // Insert a global ordering toolbar (once per page)
        try {
            let toolbar = document.getElementById('ordering-toolbar');
            if (!toolbar && elements.predictionResultsBody && elements.predictionResultsBody.parentElement) {
                toolbar = document.createElement('div');
                toolbar.id = 'ordering-toolbar';
                toolbar.className = 'd-flex justify-content-end mb-2';
                toolbar.innerHTML = `
                    <div class="input-group input-group-sm" style="max-width: 260px;">
                        <label class="input-group-text" for="ordering-select">Order by</label>
                        <select id="ordering-select" class="form-select form-select-sm">
                            <option value="win_prob">Win Probability</option>
                            <option value="predicted_rank">Predicted Rank</option>
                        </select>
                    </div>`;
                elements.predictionResultsBody.parentElement.insertBefore(toolbar, elements.predictionResultsBody);
                const orderingSelect = document.getElementById('ordering-select');
                if (orderingSelect) {
                    orderingSelect.value = window.predOrderingMode || 'win_prob';
                    if (!orderingSelect._bound) {
                        orderingSelect.addEventListener('change', () => {
                            window.predOrderingMode = orderingSelect.value || 'win_prob';
                            try { showToast(`Ordering set to: ${window.predOrderingMode === 'predicted_rank' ? 'Predicted Rank' : 'Win Probability'}`, 'info'); } catch {}
                            // Re-render any expanded details panels to reflect new ordering
                            const panels = document.querySelectorAll('.prediction-details');
                            panels.forEach((panel) => {
                                const id = panel.id || '';
                                const m = id.match(/^details-(\d+)$/);
                                if (m) {
                                    const idx = parseInt(m[1], 10);
                                    if (panel.innerHTML.trim() !== '') {
                                        panel.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Updating order...</div>';
                                        panel.removeAttribute('data-loaded');
                                        if (typeof window.__fetchAndRenderDetails === 'function') {
                                            window.__fetchAndRenderDetails(idx);
                                        }
                                    }
                                }
                            });
                        });
                        orderingSelect._bound = true;
                    }
                }
            } else if (toolbar) {
                const orderingSelect = document.getElementById('ordering-select');
                if (orderingSelect) orderingSelect.value = window.predOrderingMode || 'win_prob';
            }
        } catch (e) { console.warn('Ordering toolbar init failed', e); }
        
        // Use Array.from to ensure forEach is available on all browsers
        Array.from(results).forEach((result, index) => {
            // Normalize enhanced endpoint shape: some endpoints return { success, prediction: { predictions: [...] } }
            const predictions = Array.isArray(result.predictions)
                ? result.predictions
                : (result.prediction && Array.isArray(result.prediction.predictions) ? result.prediction.predictions : []);
            // Determine effective success: honor explicit success, otherwise infer from predictions or success-shaped message
            let success = !!result.success;
            const msgTop = (result && (result.message || result.error)) || '';
            const msgNested = (result && result.prediction && (result.prediction.message || '')) || '';
            const rawMessage = `${String(msgTop)} ${String(msgNested)}`.trim();
            const msgIndicatesCompletion = /prediction\s+completed/i.test(rawMessage);
            if (!success) {
                if ((Array.isArray(predictions) && predictions.length > 0) || msgIndicatesCompletion) {
                    success = true; // treat as successful render if payload clearly indicates completion
                }
            }
            const raceFilenameFromNested = result.prediction && result.prediction.race_info && result.prediction.race_info.filename;
            const displayRaceId = result.race_filename || result.race_id || raceFilenameFromNested || `Race ${index + 1}`;

            const resultDiv = document.createElement('div');
            const alertClass = success ? (result.degraded ? 'alert-warning' : 'alert-success') : 'alert-danger';
            resultDiv.className = `alert ${alertClass} mb-3`;
            
            if (success && predictions && predictions.length > 0) {
                const sortedPreds = Array.isArray(predictions)
                    ? [...predictions].sort((a, b) => Number(b.win_prob || b.normalized_win_probability || b.final_score || b.prediction_score || b.win_probability || b.confidence || 0) - Number(a.win_prob || a.normalized_win_probability || a.final_score || a.prediction_score || a.win_probability || a.confidence || 0))
                    : [];
                const topPick = result.top_pick || (sortedPreds.length ? sortedPreds[0] : (predictions ? predictions[0] : null));
                
                if (topPick) {
                    const winProb = Number(topPick.win_prob || topPick.normalized_win_probability || topPick.final_score || topPick.win_probability || topPick.confidence || 0);
                    const dogName = topPick.dog_name || topPick.name || 'Unknown';
                    const raceInfo = displayRaceId;

                    // Build an encoded name for direct API link (strip .csv)
                    const linkNameRaw = (result.race_filename || (raceFilenameFromNested || '')).replace(/\.csv$/i, '') || (result.race_id || '');
                    const encodedName = encodeURIComponent(String(linkNameRaw));
                    const apiHref = encodedName ? `/api/prediction_detail/${encodedName}` : '#';
                    
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
                                    ${encodedName ? ` | <a class="link-secondary" href="${apiHref}" target="_blank" rel="noopener noreferrer"><i class="fas fa-file-code"></i> Raw JSON</a>` : ''}
                                    | <a href="#" class="link-primary" onclick="return __ensureExpandAndRender(${index});"><i class="fas fa-eye"></i> View Details</a>
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
                // Prefer nested message if present; otherwise show top-level
                const errorMessage = (result && (result.message || result.error)) || 'Unknown error occurred';
                const raceInfo = displayRaceId;
                
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

            // Auto-fetch and render details for each successful result to avoid relying on click binding timing
            if (success) {
                (async () => {
                    try {
                        const btn = resultDiv.querySelector('button[data-expanded]');
                        const detailsDiv = resultDiv.querySelector(`#details-${index}`);
                        if (!btn || !detailsDiv) return;
                        // If already populated, skip
                        if (detailsDiv.innerHTML.trim() !== '') return;
                        // Reuse toggleDetails logic by simulating expand state
                        await window.__fetchAndRenderDetails(index);
                        // Set expanded UI state
                        detailsDiv.style.display = 'block';
                        btn.setAttribute('data-expanded', 'true');
                        const icon = btn.querySelector('i');
                        if (icon) icon.className = 'fas fa-chevron-up';
                        btn.innerHTML = '<i class="fas fa-chevron-up"></i> Hide';
                    } catch (e) {
                        console.warn('Auto details render failed', e);
                    }
                })();
            }
        });
        
        // Add summary at the bottom (use effective success evaluation)
        const successCount = results.filter(r => {
            const preds = Array.isArray(r.predictions)
                ? r.predictions
                : (r.prediction && Array.isArray(r.prediction.predictions) ? r.prediction.predictions : []);
            const msgTop = (r && (r.message || r.error)) || '';
            const msgNested = (r && r.prediction && (r.prediction.message || '')) || '';
            const rawMsg = `${String(msgTop)} ${String(msgNested)}`.trim();
            const completion = /prediction\s+completed/i.test(rawMsg);
            return !!r.success || (Array.isArray(preds) && preds.length > 0) || completion;
        }).length;
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
    
    // Ensure details panel is expanded and rendered (safe public helper)
    window.__ensureExpandAndRender = async function(index) {
        try {
            const detailsDiv = document.getElementById(`details-${index}`);
            if (detailsDiv) {
                detailsDiv.style.display = 'block';
            }
            await window.__fetchAndRenderDetails(index);
        } catch (e) {
            console.warn('ensureExpandAndRender failed', e);
            try { showToast('Failed to load details', 'warning'); } catch {}
        }
        return false;
    };

    // Internal helper to fetch and render details into the details div (id resolves via index)
    window.__fetchAndRenderDetails = async function(index) {
        const detailsDiv = document.getElementById(`details-${index}`);
        if (!detailsDiv) return;
        // Avoid double-loading
        if (detailsDiv.getAttribute('data-loaded') === 'true' && detailsDiv.innerHTML.trim() !== '') return;
        detailsDiv.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading details...</div>';

        const resultIndex = parseInt(index);
        const currentResults = window.currentPredictionResults || [];
        const detailsResult = currentResults[resultIndex];

        try {
            const nestedFilename = detailsResult && detailsResult.prediction && detailsResult.prediction.race_info && detailsResult.prediction.race_info.filename;
            const rawName = (detailsResult && (detailsResult.race_filename || (detailsResult.race_id || '') )) || (nestedFilename || '');
            let raceNameForApi = String(rawName)
                .replace(/\.csv$/i, '')
                .replace(/^prediction_/, '')
                .replace(/\.json$/i, '');

            if (raceNameForApi) {
                const resp = await fetchWithErrorHandling(`/api/prediction_detail/${encodeURIComponent(raceNameForApi)}`);
                const data = await resp.json();
                if (data && data.success && data.prediction) {
                    const pred = data.prediction;
                    const raceCtx = pred.race_context || {};
                    const enhanced = pred.enhanced_predictions || pred.predictions || [];
                    const enhancedSorted = Array.isArray(enhanced)
                        ? sortPreds(enhanced, window.predOrderingMode || 'win_prob')
                        : [];
                    const topPick = pred.top_pick || (enhancedSorted.length ? enhancedSorted[0] : (enhanced.length ? enhanced[0] : null));

                    let detailsHTML = '';
                    // Summary header with actions
                    const rawApiUrl = `/api/prediction_detail/${encodeURIComponent(raceNameForApi)}`;
                    const downloadJsonName = `prediction_${raceNameForApi}.json`;
                    const downloadJsonHref = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(pred, null, 2))}`;
                    detailsHTML += `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                          <div><strong>Venue:</strong> ${raceCtx.venue || 'Unknown'} | <strong>Date:</strong> ${raceCtx.race_date || 'Unknown'} | <strong>Distance:</strong> ${raceCtx.distance || 'Unknown'} | <strong>Grade:</strong> ${raceCtx.grade || 'Unknown'}</div>
                          <div class="d-flex gap-2">
                            <a class="btn btn-sm btn-outline-secondary" href="${rawApiUrl}" target="_blank" rel="noopener noreferrer"><i class="fas fa-file-code"></i> View raw JSON (API)</a>
                            <a class="btn btn-sm btn-outline-secondary" href="${downloadJsonHref}" download="${downloadJsonName}"><i class="fas fa-download"></i> Download JSON</a>
                          </div>
                        </div>`;

                    // Technical summary of prediction pipeline (enhanced visibility)
                    try {
                        const predictor = pred.predictor_used || pred.model_used || 'Unknown';
                        const methodsArr = Array.isArray(pred.prediction_methods_used) ? pred.prediction_methods_used : (pred.prediction_method ? [pred.prediction_method] : []);
                        const methodsStr = methodsArr.length ? methodsArr.join(', ') : 'Unknown';
                        const analysisVersion = pred.analysis_version || pred.version || 'Unknown';
                        const timestamp = pred.prediction_timestamp || pred.timestamp || '';
                        const totalDogs = Array.isArray(enhanced) ? enhanced.length : (Array.isArray(pred.predictions) ? pred.predictions.length : 0);
                        const dataSources = (() => {
                            try {
                                const src = pred.data_sources || pred.sources || {};
                                if (Array.isArray(src)) return src.join(', ');
                                if (src && typeof src === 'object') return Object.keys(src).join(', ');
                            } catch {}
                            return '';
                        })();

                        detailsHTML += `
                          <div class="card mb-2">
                            <div class="card-body p-2">
                              <div class="row g-2 small">
                                <div class="col-md-6"><strong>Predictor:</strong> ${predictor}</div>
                                <div class="col-md-6"><strong>Methods:</strong> ${methodsStr}</div>
                                <div class="col-md-6"><strong>Analysis Version:</strong> ${analysisVersion}</div>
                                <div class="col-md-6"><strong>Prediction Time:</strong> ${timestamp || 'N/A'}</div>
                                <div class="col-md-6"><strong>Dogs Analyzed:</strong> ${totalDogs}</div>
                                ${dataSources ? `<div class="col-12"><strong>Data Sources:</strong> ${dataSources}</div>` : ''}
                              </div>
                            </div>
                          </div>`;
                    } catch (e) {
                        console.warn('Failed to render technical summary', e);
                    }

                    if (topPick) {
                        let tpName = topPick.dog_name || topPick.clean_name || 'Unknown';
                        let tpScore = Number(topPick.win_prob || topPick.normalized_win_probability || topPick.final_score || topPick.prediction_score || topPick.win_probability || topPick.confidence || 0);
                        let tpBox = topPick.box_number || topPick.box || 'N/A';
                        // Fallback: if no numeric score on top_pick, derive from first runner
                        if ((!isFinite(tpScore) || tpScore === 0) && Array.isArray(enhanced) && enhanced.length) {
                            const first = enhanced[0];
                            tpName = tpName !== 'Unknown' ? tpName : (first.dog_name || first.clean_name || tpName);
                            tpScore = Number(first.win_prob || first.normalized_win_probability || first.final_score || first.prediction_score || first.win_probability || first.confidence || tpScore);
                            tpBox = tpBox !== 'N/A' ? tpBox : (first.box_number || first.box || tpBox);
                        }
                        detailsHTML += `
                          <div class="card mb-2">
                            <div class="card-body p-2">
                              <h6 class="mb-1"><i class="fas fa-trophy text-warning"></i> Top Pick: ${tpName}</h6>
                              <small>Box: ${tpBox} | Confidence: ${(tpScore*100).toFixed(1)}%</small>
                              ${Array.isArray(topPick.key_factors) && topPick.key_factors.length ? `<div class="mt-2"><small><strong>Key factors:</strong> ${topPick.key_factors.slice(0,5).join('; ')}</small></div>` : ''}
                            </div>
                          </div>`;
                    }

                    if (Array.isArray(enhancedSorted) && enhancedSorted.length) {
                        detailsHTML += '<div class="row">';
                        enhancedSorted.forEach((d, idx) => {
                            const name = d.dog_name || d.clean_name || 'Unknown';
                            const score = Number(d.win_prob || d.normalized_win_probability || d.final_score || d.prediction_score || d.win_probability || d.confidence || 0);
                            const box = d.box_number || d.box || 'N/A';
                            const extra = Array.isArray(d.key_factors) && d.key_factors.length ? `<div class="mt-1"><small>${d.key_factors.slice(0,3).join(' • ')}</small></div>` : '';
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
                    }

                    // Optional advisory
                    try {
                        const advResp = await fetchWithErrorHandling('/api/generate_advisory', {
                            method: 'POST',
                            body: JSON.stringify({ prediction_data: pred })
                        });
                        const advData = await advResp.json();
                        if (advData) {
                            const advisory = advData.advisory || advData.enhancement || advData;
                            const getSummary = (obj) => {
                                try {
                                    if (typeof obj === 'string') return obj.split('\n')[0].slice(0, 200);
                                    if (obj && typeof obj === 'object') {
                                        if (obj.summary) return String(obj.summary).slice(0, 200);
                                        if (Array.isArray(obj.bullets) && obj.bullets.length) return String(obj.bullets[0]).slice(0, 200);
                                        if (obj.message) return String(obj.message).slice(0, 200);
                                        // Suppress JSON fragments in compact banner
                                        return '';
                                    }
                                } catch {}
                                return '';
                            };
                            const summaryText = getSummary(advisory);
                            if (summaryText) {
                                detailsHTML = `\n<div class="alert alert-info py-1 px-2 mb-2"><i class="fas fa-lightbulb me-1"></i>${summaryText}</div>` + detailsHTML;
                            }
                        }
                    } catch {}

                    detailsDiv.innerHTML = detailsHTML || '<p class="text-muted">No detailed prediction data available.</p>';
                    detailsDiv.setAttribute('data-loaded', 'true');
                    try { console.log('Details loaded for index', index); showToast('Details loaded', 'success'); } catch {}
                    return;
                }
            }
        } catch (e) {
            console.warn('Rich prediction details fetch failed, falling back to minimal render', e);
        }

        // Fallback minimal render
        const currentResults2 = window.currentPredictionResults || [];
        const resultFallback = currentResults2[index];
        if (resultFallback && resultFallback.predictions && resultFallback.predictions.length > 0) {
            let detailsHTML = '<div class="row">';
            const sortedPredsFallback = sortPreds(resultFallback.predictions, window.predOrderingMode || 'win_prob');
            Array.from(sortedPredsFallback).forEach((prediction, idx) => {
                const winProb = Number(prediction.win_prob || prediction.normalized_win_probability || prediction.final_score || prediction.prediction_score || prediction.win_probability || prediction.confidence || 0);
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
            if (resultFallback.message) {
                detailsHTML += `<div class="mt-2"><small class="text-muted"><i class="fas fa-info-circle"></i> ${resultFallback.message}</small></div>`;
            }
            detailsDiv.innerHTML = detailsHTML;
            detailsDiv.setAttribute('data-loaded', 'true');
        } else {
            detailsDiv.innerHTML = '<p class="text-muted">No detailed prediction data available.</p>';
            detailsDiv.setAttribute('data-loaded', 'true');
        }
    };

    // Toggle prediction details
    window.toggleDetails = async function(button, index) {
        const detailsDiv = document.getElementById(`details-${index}`);
        const icon = button.querySelector('i');
        const isExpanded = button.getAttribute('data-expanded') === 'true';
        
        if (isExpanded) {
            detailsDiv.style.display = 'none';
            icon.className = 'fas fa-chevron-down';
            button.innerHTML = '<i class="fas fa-chevron-down"></i> Details';
            button.setAttribute('data-expanded', 'false');
            return;
        }

        // Expand
        detailsDiv.style.display = 'block';
        icon.className = 'fas fa-chevron-up';
        button.innerHTML = '<i class="fas fa-chevron-up"></i> Hide';
        button.setAttribute('data-expanded', 'true');
        
        // Load detailed information if not already loaded
        if (detailsDiv.innerHTML.trim() !== '') {
            return;
        }

        detailsDiv.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading details...</div>';
        
        // Get the result data from the current results array
        const resultIndex = parseInt(index);
        const currentResults = window.currentPredictionResults || [];
        const detailResult = currentResults[resultIndex];

        // If we have a race filename, try to fetch rich details from the backend
        // Fallback to local minimal render otherwise
        try {
            const nestedFilename = detailResult && detailResult.prediction && detailResult.prediction.race_info && detailResult.prediction.race_info.filename;
            const rawName = (detailResult && (detailResult.race_filename || (detailResult.race_id || '') ) ) || (nestedFilename || '');
            let raceNameForApi = String(rawName)
                .replace(/\.csv$/i, '')
                .replace(/^prediction_/, '')
                .replace(/\.json$/i, '');

            if (raceNameForApi) {
                const resp = await fetchWithErrorHandling(`/api/prediction_detail/${encodeURIComponent(raceNameForApi)}`);
                const data = await resp.json();
                if (data && data.success && data.prediction) {
                    const pred = data.prediction;
                    const raceCtx = pred.race_context || {};
                    const enhanced = pred.enhanced_predictions || pred.predictions || [];
                    const enhancedSorted = Array.isArray(enhanced)
                        ? sortPreds(enhanced, window.predOrderingMode || 'win_prob')
                        : [];
                    const topPick = pred.top_pick || (enhancedSorted.length ? enhancedSorted[0] : null);

                    let detailsHTML = '';
                    // Summary header with actions
                    const rawApiUrl = `/api/prediction_detail/${encodeURIComponent(raceNameForApi)}`;
                    const downloadJsonName = `prediction_${raceNameForApi}.json`;
                    const downloadJsonHref = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(pred, null, 2))}`;
                    detailsHTML += `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                          <div><strong>Venue:</strong> ${raceCtx.venue || 'Unknown'} | <strong>Date:</strong> ${raceCtx.race_date || 'Unknown'} | <strong>Distance:</strong> ${raceCtx.distance || 'Unknown'} | <strong>Grade:</strong> ${raceCtx.grade || 'Unknown'}</div>
                          <div class="d-flex gap-2">
                            <a class="btn btn-sm btn-outline-secondary" href="${rawApiUrl}" target="_blank" rel="noopener noreferrer"><i class="fas fa-file-code"></i> View raw JSON (API)</a>
                            <a class="btn btn-sm btn-outline-secondary" href="${downloadJsonHref}" download="${downloadJsonName}"><i class="fas fa-download"></i> Download JSON</a>
                          </div>
                        </div>`;

                    // Technical summary of prediction pipeline (enhanced visibility)
                    try {
                        const predictor = pred.predictor_used || pred.model_used || 'Unknown';
                        const methodsArr = Array.isArray(pred.prediction_methods_used) ? pred.prediction_methods_used : (pred.prediction_method ? [pred.prediction_method] : []);
                        const methodsStr = methodsArr.length ? methodsArr.join(', ') : 'Unknown';
                        const analysisVersion = pred.analysis_version || pred.version || 'Unknown';
                        const timestamp = pred.prediction_timestamp || pred.timestamp || '';
                        const totalDogs = Array.isArray(enhanced) ? enhanced.length : (Array.isArray(pred.predictions) ? pred.predictions.length : 0);
                        const dataSources = (() => {
                            try {
                                const src = pred.data_sources || pred.sources || {};
                                if (Array.isArray(src)) return src.join(', ');
                                if (src && typeof src === 'object') return Object.keys(src).join(', ');
                            } catch {}
                            return '';
                        })();

                        detailsHTML += `
                          <div class="card mb-2">
                            <div class="card-body p-2">
                              <div class="row g-2 small">
                                <div class="col-md-6"><strong>Predictor:</strong> ${predictor}</div>
                                <div class="col-md-6"><strong>Methods:</strong> ${methodsStr}</div>
                                <div class="col-md-6"><strong>Analysis Version:</strong> ${analysisVersion}</div>
                                <div class="col-md-6"><strong>Prediction Time:</strong> ${timestamp || 'N/A'}</div>
                                <div class="col-md-6"><strong>Dogs Analyzed:</strong> ${totalDogs}</div>
                                ${dataSources ? `<div class="col-12"><strong>Data Sources:</strong> ${dataSources}</div>` : ''}
                              </div>
                            </div>
                          </div>`;
                    } catch (e) {
                        console.warn('Failed to render technical summary', e);
                    }

                    // Top pick card
                    if (topPick) {
                        let tpName = topPick.dog_name || topPick.clean_name || 'Unknown';
                        let tpScore = Number(topPick.win_prob || topPick.normalized_win_probability || topPick.final_score || topPick.prediction_score || topPick.win_probability || topPick.confidence || 0);
                        let tpBox = topPick.box_number || topPick.box || 'N/A';
                        if ((!isFinite(tpScore) || tpScore === 0) && Array.isArray(enhanced) && enhanced.length) {
                            const first = enhanced[0];
                            tpName = tpName !== 'Unknown' ? tpName : (first.dog_name || first.clean_name || tpName);
                            tpScore = Number(first.win_prob || first.normalized_win_probability || first.final_score || first.prediction_score || first.win_probability || first.confidence || tpScore);
                            tpBox = tpBox !== 'N/A' ? tpBox : (first.box_number || first.box || tpBox);
                        }
                        detailsHTML += `
                          <div class="card mb-2">
                            <div class="card-body p-2">
                              <h6 class="mb-1"><i class="fas fa-trophy text-warning"></i> Top Pick: ${tpName}</h6>
                              <small>Box: ${tpBox} | Confidence: ${(tpScore*100).toFixed(1)}%</small>
                              ${Array.isArray(topPick.key_factors) && topPick.key_factors.length ? `<div class="mt-2"><small><strong>Key factors:</strong> ${topPick.key_factors.slice(0,5).join('; ')}</small></div>` : ''}
                            </div>
                          </div>`;
                    }

                    // Runners grid
                    if (Array.isArray(enhancedSorted) && enhancedSorted.length) {
                        detailsHTML += '<div class="row">';
                        enhancedSorted.forEach((d, idx) => {
                            const name = d.dog_name || d.clean_name || 'Unknown';
                            const score = Number(d.win_prob || d.normalized_win_probability || d.final_score || d.prediction_score || d.win_probability || d.confidence || 0);
                            const box = d.box_number || d.box || 'N/A';
                            const extra = Array.isArray(d.key_factors) && d.key_factors.length ? `<div class="mt-1"><small>${d.key_factors.slice(0,3).join(' • ')}</small></div>` : '';
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
                    }

                    // Try to fetch advisory (compact summary + full card)
                    try {
                        const advResp = await fetchWithErrorHandling('/api/generate_advisory', {
                            method: 'POST',
                            body: JSON.stringify({ prediction_data: pred })
                        });
                        const advData = await advResp.json();
                        if (advData) {
                            const advisory = advData.advisory || advData.enhancement || advData;
                            const getSummary = (obj) => {
                                try {
                                    if (typeof obj === 'string') return obj.split('\n')[0].slice(0, 200);
                                    if (obj && typeof obj === 'object') {
                                        if (obj.summary) return String(obj.summary).slice(0, 200);
                                        if (Array.isArray(obj.bullets) && obj.bullets.length) return String(obj.bullets[0]).slice(0, 200);
                                        if (obj.message) return String(obj.message).slice(0, 200);
                                        // Suppress JSON fragments in compact banner
                                        return '';
                                    }
                                } catch {}
                                return '';
                            };
                            const summaryText = getSummary(advisory);
                            if (summaryText) {
                                detailsHTML = `\n<div class="alert alert-info py-1 px-2 mb-2"><i class="fas fa-lightbulb me-1"></i>${summaryText}</div>` + detailsHTML;
                            }
                            const pretty = (() => { try { return JSON.stringify(advisory, null, 2); } catch { return String(advisory); } })();
                            detailsHTML += `
                                <div class="card mt-2">
                                  <div class="card-header p-2"><i class="fas fa-lightbulb"></i> Advisory</div>
                                  <div class="card-body p-2">
                                    <pre class="mb-0" style="white-space: pre-wrap;">${pretty}</pre>
                                  </div>
                                </div>`;
                        }
                    } catch (e) {
                        console.warn('Advisory fetch failed', e);
                    }

                    detailsDiv.innerHTML = detailsHTML || '<p class="text-muted">No detailed prediction data available.</p>';
                    return;
                }
            }
        } catch (e) {
            console.warn('Rich prediction details fetch failed, falling back to minimal render', e);
            try { showToast('Failed to load rich details; showing minimal info', 'warning'); } catch {}
        }

        // Fallback: minimal render from in-memory result
        if (detailResult && detailResult.predictions && detailResult.predictions.length > 0) {
            let detailsHTML = '<div class="row">';
            const sortedPredsFallback2 = sortPreds(detailResult.predictions, window.predOrderingMode || 'win_prob');
            Array.from(sortedPredsFallback2).forEach((prediction, idx) => {
                const winProb = Number(prediction.win_prob || prediction.normalized_win_probability || prediction.final_score || prediction.prediction_score || prediction.win_probability || prediction.confidence || 0);
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
            if (detailResult.message) {
                detailsHTML += `<div class="mt-2"><small class="text-muted"><i class="fas fa-info-circle"></i> ${detailResult.message}</small></div>`;
            }
            detailsDiv.innerHTML = detailsHTML;
        } else {
            detailsDiv.innerHTML = '<p class="text-muted">No detailed prediction data available.</p>';
        }
    };

    // Utility to show alerts with proper container handling and graceful fallback
    function showAlert(message, type = 'info') {
        // Try to find the actual alert container present in the DOM
        let container = document.getElementById('alertContainer') ||
                       document.getElementById('alert-container') ||
                       document.getElementById('alerts-container') ||
                       document.getElementById('notification-container');
        
        // Graceful fallback: create container if none exists
        if (!container) {
            container = document.createElement('div');
            container.id = 'alertContainer';
            container.className = 'position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1055';
            container.setAttribute('role', 'alert');
            container.setAttribute('aria-live', 'polite');
            document.body.appendChild(container);
            console.log('✅ Alert container created with ID: alertContainer');
        }
        
        // Try to use ErrorDisplayManager if available
        if (typeof window.errorManager !== 'undefined' && window.errorManager.showAlert) {
            return window.errorManager.showAlert(message, type);
        }
        
        // Fallback implementation using the found/created container
        const alertId = `alert-${Date.now()}`;
        const alert = document.createElement('div');
        alert.id = alertId;
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.setAttribute('role', 'alert');
        
        const iconMap = {
            danger: '<i class="fas fa-exclamation-circle me-2"></i>',
            warning: '<i class="fas fa-exclamation-triangle me-2"></i>',
            info: '<i class="fas fa-info-circle me-2"></i>',
            success: '<i class="fas fa-check-circle me-2"></i>'
        };
        
        alert.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="flex-grow-1">
                    ${iconMap[type] || ''}
                    ${message}
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        container.appendChild(alert);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement && typeof bootstrap !== 'undefined' && bootstrap.Alert) {
                const bsAlert = new bootstrap.Alert(alertElement);
                bsAlert.close();
            } else if (alertElement) {
                alertElement.remove();
            }
        }, 5000);
        
        return alertId;
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
    
    // Check for missing elements in second DOMContentLoaded block
    const missingElements = [];
    // Use Array.from to ensure forEach is available on all browsers
    Array.from(Object.keys(elements)).forEach(key => {
        if (!elements[key]) {
            missingElements.push(key);
            console.warn(`Missing DOM element in second block: ${key}`);
        }
    });
    
    if (missingElements.length > 0) {
        // If the primary container for this second block isn't present, this page doesn't use it.
        // Quietly exit to avoid noisy warnings on pages that only use the first block/table view.
        if (!elements.racesContainer) {
            return; // no-op on this page
        }
        // Otherwise log once for debugging
        console.warn('Second Interactive Races block: Some elements are missing:', missingElements);
    }

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

    // Venue display name mapper for common codes
    const VENUE_DISPLAY = {
        'AP_K': 'Angle Park',
        'APTH': 'Albion Park',
        'APWE': 'Albion Park',
        'BAL': 'Ballarat',
        'BEN': 'Bendigo',
        'CANN': 'Cannington',
        'CASO': 'Casino',
        'DAPT': 'Dapto',
        'GEE': 'Geelong',
        'GAWL': 'Gawler',
        'GRDN': 'The Gardens',
        'HEA': 'Healesville',
        'HOBT': 'Hobart',
        'HOR': 'Horsham',
        'MAND': 'Mandurah',
        'MEA': 'The Meadows',
        'MOUNT': 'Mount Gambier',
        'MURR': 'Murray Bridge',
        'NOR': 'Northam',
        'RICH': 'Richmond',
        'SAL': 'Sale',
        'SAN': 'Sandown Park',
        'SHEP': 'Shepparton',
        'TRA': 'Traralgon',
        'WAR': 'Warrnambool',
        'W_PK': 'Wentworth Park',
        'WPK': 'Wentworth Park'
    };

    function prettyVenue(venue) {
        if (!venue) return 'Unknown';
        const key = String(venue).toUpperCase().trim();
        return VENUE_DISPLAY[key] || venue;
    }

    function renderRaces() {
        if (!elements.racesContainer) return;
        elements.racesContainer.innerHTML = '';

        if (!state.races || state.races.length === 0) {
            if (elements.noResultsMessage) elements.noResultsMessage.style.display = 'block';
            return;
        }
        if (elements.noResultsMessage) elements.noResultsMessage.style.display = 'none';

        state.races.forEach(race => {
            const card = document.createElement('div');
            card.className = 'race-card expanded'; // expanded by default

            const venueName = prettyVenue(race.venue);
            const raceTitle = `${venueName} - Race ${race.race_number}`;
            const meta = `${race.race_date || ''} | ${race.distance || ''} | ${race.grade || ''}`;

            // Sort and filter runners if present
            const runners = Array.isArray(race.runners) ? race.runners.slice() : [];
            const validRunners = runners.filter(r => r && r.dog_name && r.dog_name.toLowerCase() !== 'vacant' && r.dog_name.toLowerCase() !== 'empty');
            validRunners.sort((a,b) => (a.predicted_rank || 99) - (b.predicted_rank || 99));

            const runnersHtml = validRunners.map(r => {
                const winProb = Math.max(0, Math.min(1, Number(r.win_probability || r.confidence || 0)));
                const placeProb = Math.max(0, Math.min(1, Number(r.place_probability || (winProb * 1.8 > 1 ? 1 : winProb * 1.8))));
                const odds = r.odds || r.odds_decimal || '';
                return `
                    <li class="runner-entry">
                        <div class="runner-number">${r.box_number || r.box || ''}</div>
                        <div class="runner-info">
                            <div class="runner-name">${r.dog_name || 'Unknown'}<span class="runner-odds">${odds ? '$' + odds : ''}</span></div>
                            <div>
                                <div class="win-bar" style="width: ${(winProb*100).toFixed(1)}%;" title="Win: ${(winProb*100).toFixed(1)}%"></div>
                                <div class="place-bar" style="width: ${(placeProb*100).toFixed(1)}%;" title="Place: ${(placeProb*100).toFixed(1)}%"></div>
                            </div>
                        </div>
                    </li>`;
            }).join('');

            card.innerHTML = `
                <div class="race-card-header">
                    <div>
                        <div class="race-title">${raceTitle}</div>
                        <div class="race-meta">${meta}</div>
                    </div>
                    <button class="btn-expand" aria-label="Toggle details">Details</button>
                </div>
                <div class="race-card-body">
                    <ul class="runners-list">${runnersHtml}</ul>
                </div>
            `;

            // toggle behavior (kept but default expanded)
            const header = card.querySelector('.race-card-header');
            header.addEventListener('click', () => {
                card.classList.toggle('expanded');
            });

            elements.racesContainer.appendChild(card);
        });
    }

    // Fetch races from backend API and then render
    async function fetchRaces() {
        try {
            elements.isLoading = true;
            if (elements.loadingSpinner) elements.loadingSpinner.style.display = 'block';

            const { sortBy, order, page, perPage, searchQuery } = state.filters;
            const url = `/api/races/paginated?sort_by=${encodeURIComponent(sortBy)}&order=${encodeURIComponent(order)}&page=${page}&per_page=${perPage}&search=${encodeURIComponent(searchQuery || '')}`;
            const resp = await fetchWithErrorHandling(url, { method: 'GET' });
            const data = await resp.json();

            state.races = Array.isArray(data.races) ? data.races : [];
            state.pagination = data.pagination || {};
            renderRaces();
        } catch (e) {
            console.warn('Failed to load races', e);
            state.races = [];
            renderRaces();
            showToast('Failed to load races. Please try again.', 'warning');
        } finally {
            elements.isLoading = false;
            if (elements.loadingSpinner) elements.loadingSpinner.style.display = 'none';
        }
    }

    function setupEventListeners() {
        if (elements.searchInput) {
            elements.searchInput.addEventListener('input', debounce(handleSearch, 300));
        }
        if (elements.clearSearch) {
            elements.clearSearch.addEventListener('click', clearSearch);
        }
        if (elements.sortSelect) {
            elements.sortSelect.addEventListener('change', handleFilterChange);
        }
        if (elements.statusFilter) {
            elements.statusFilter.addEventListener('change', handleFilterChange);
        }
        if (elements.venueFilter) {
            elements.venueFilter.addEventListener('change', handleFilterChange);
        }
        if (elements.toggleView) {
            elements.toggleView.addEventListener('click', toggleView);
        }
    }

    async function loadVenuesAndGrades() {
        // In a real application, you would fetch these from an API endpoint
        state.venues = ['The Meadows', 'Sandown Park', 'Wentworth Park', 'Albion Park'];
        state.grades = ['Group 1', 'Group 2', 'Group 3', 'Maiden', 'Novice'];
        
        populateSelect(elements.venueFilter, state.venues, 'All Venues');
    }

    function populateSelect(selectElement, options, defaultOption) {
        if (!selectElement) {
            console.warn('Cannot populate select - element is null');
            return;
        }
        selectElement.innerHTML = `<option value="all">${defaultOption}</option>`;
        options.forEach(option => {
            selectElement.innerHTML += `<option value="${option}">${option}</option>`;
        });
    }

    function handleSearch(event) {
        state.filters.searchQuery = event.target ? event.target.value : '';
        state.filters.page = 1;
        fetchRaces();
    }

    function clearSearch() {
        if (elements.searchInput) {
            elements.searchInput.value = '';
        }
        state.filters.searchQuery = '';
        state.filters.page = 1;
        fetchRaces();
    }

    function handleFilterChange() {
        if (elements.sortSelect && elements.sortSelect.value) {
            const sortValue = elements.sortSelect.value.split('|');
            state.filters.sortBy = sortValue[0];
            state.filters.order = sortValue[1];
        }
        if (elements.statusFilter) {
            state.filters.status = elements.statusFilter.value;
        }
        if (elements.venueFilter) {
            state.filters.venue = elements.venueFilter.value;
        }
        state.filters.page = 1;
        fetchRaces();
    }

    function toggleView() {
        state.view = state.view === 'grid' ? 'list' : 'grid';
        if (elements.viewIcon) {
            elements.viewIcon.className = state.view === 'grid' ? 'fas fa-th-large' : 'fas fa-list';
        }
        if (elements.racesContainer) {
            elements.racesContainer.className = `races-container ${state.view}`;
        }
    }

    init();
});

