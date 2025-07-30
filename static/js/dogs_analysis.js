// Global variables for pagination
let currentPage = 1;
let currentSortBy = 'total_races';
let currentOrder = 'desc';
let currentPerPage = 50;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the page
    loadTopPerformers();
    
    // Set up search functionality
    const searchInput = document.getElementById('dog-search-input');
    const searchBtn = document.getElementById('dog-search-btn');
    
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    // Set up all dogs functionality
    const loadAllDogsBtn = document.getElementById('load-all-dogs-btn');
    const sortSelect = document.getElementById('dogs-sort-select');
    const orderSelect = document.getElementById('dogs-order-select');
    const perPageSelect = document.getElementById('dogs-per-page-select');
    
    loadAllDogsBtn.addEventListener('click', loadAllDogs);
    sortSelect.addEventListener('change', function() {
        currentSortBy = this.value;
        currentPage = 1; // Reset to first page
        loadAllDogs();
    });
    orderSelect.addEventListener('change', function() {
        currentOrder = this.value;
        currentPage = 1; // Reset to first page
        loadAllDogs();
    });
    perPageSelect.addEventListener('change', function() {
        currentPerPage = parseInt(this.value);
        currentPage = 1; // Reset to first page
        loadAllDogs();
    });
});

function performSearch() {
    const searchTerm = document.getElementById('dog-search-input').value.trim();
    if (!searchTerm) {
        alert('Please enter a dog name to search');
        return;
    }
    
    const resultsContainer = document.getElementById('search-results-container');
    resultsContainer.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
    
    fetch(`/api/dogs/search?q=${encodeURIComponent(searchTerm)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displaySearchResults(data.dogs);
            } else {
                resultsContainer.innerHTML = `<div class="alert alert-warning">No dogs found matching "${searchTerm}"</div>`;
            }
        })
        .catch(error => {
            console.error('Search error:', error);
            resultsContainer.innerHTML = '<div class="alert alert-danger">Error performing search. Please try again.</div>';
        });
}

function displaySearchResults(dogs) {
    const resultsContainer = document.getElementById('search-results-container');
    
    if (!dogs || dogs.length === 0) {
        resultsContainer.innerHTML = '<div class="alert alert-info">No dogs found.</div>';
        return;
    }
    
    let html = '<div class="row">';
    
    dogs.forEach(dog => {
        const winRate = dog.total_races > 0 ? ((dog.total_wins / dog.total_races) * 100).toFixed(1) : '0.0';
        const placeRate = dog.total_races > 0 ? ((dog.total_places / dog.total_races) * 100).toFixed(1) : '0.0';
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">${dog.dog_name}</h5>
                        <div class="row">
                            <div class="col-6">
                                <small class="text-muted">Total Races:</small><br>
                                <strong>${dog.total_races}</strong>
                            </div>
                            <div class="col-6">
                                <small class="text-muted">Wins:</small><br>
                                <strong>${dog.total_wins}</strong>
                            </div>
                        </div>
                        <div class="row mt-2">
                            <div class="col-6">
                                <small class="text-muted">Win Rate:</small><br>
                                <span class="badge ${winRate > 20 ? 'bg-success' : winRate > 10 ? 'bg-warning' : 'bg-secondary'}">${winRate}%</span>
                            </div>
                            <div class="col-6">
                                <small class="text-muted">Place Rate:</small><br>
                                <span class="badge ${placeRate > 40 ? 'bg-success' : placeRate > 25 ? 'bg-warning' : 'bg-secondary'}">${placeRate}%</span>
                            </div>
                        </div>
                        ${dog.best_time ? `
                        <div class="mt-2">
                            <small class="text-muted">Best Time:</small><br>
                            <strong>${dog.best_time}s</strong>
                        </div>
                        ` : ''}
                        ${dog.last_race_date ? `
                        <div class="mt-2">
                            <small class="text-muted">Last Race:</small><br>
                            <small>${new Date(dog.last_race_date).toLocaleDateString()}</small>
                        </div>
                        ` : ''}
                        <div class="mt-3">
                            <button class="btn btn-primary btn-sm" onclick="viewDogDetails('${dog.dog_name}')">
                                <i class="fas fa-eye"></i> View Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    resultsContainer.innerHTML = html;
}

function loadTopPerformers() {
    const container = document.getElementById('top-performers-container');
    container.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';
    
    fetch('/api/dogs/top_performers?metric=win_rate&limit=5&min_races=1')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.top_performers && data.top_performers.length > 0) {
                displayTopPerformers(data.top_performers);
            } else {
                container.innerHTML = '<div class="text-muted">No top performers data available.</div>';
            }
        })
        .catch(error => {
            console.error('Error loading top performers:', error);
            container.innerHTML = '<div class="text-danger">Error loading top performers.</div>';
        });
}

function displayTopPerformers(dogs) {
    const container = document.getElementById('top-performers-container');
    
    let html = '<div class="list-group list-group-flush">';
    
    dogs.forEach((dog, index) => {
        const winRate = dog.total_races > 0 ? ((dog.total_wins / dog.total_races) * 100).toFixed(1) : '0.0';
        
        html += `
            <div class="list-group-item d-flex justify-content-between align-items-center px-0">
                <div>
                    <div class="fw-bold">${index + 1}. ${dog.dog_name}</div>
                    <small class="text-muted">${dog.total_races} races, ${dog.total_wins} wins</small>
                </div>
                <div class="text-end">
                    <span class="badge bg-success">${winRate}%</span>
                    <br>
                    <button class="btn btn-sm btn-outline-primary mt-1" onclick="viewDogDetails('${dog.dog_name}')">
                        Details
                    </button>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function viewDogDetails(dogName) {
    const modal = new bootstrap.Modal(document.getElementById('dog-details-modal'));
    const modalTitle = document.getElementById('dog-details-modal-title');
    const modalBody = document.getElementById('dog-details-modal-body');
    
    modalTitle.textContent = `${dogName} - Details`;
    modalBody.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading details...</div>';
    
    modal.show();
    
    // Load dog details and form guide
    Promise.all([
        fetch(`/api/dogs/${encodeURIComponent(dogName)}/details`).then(r => r.json()),
        fetch(`/api/dogs/${encodeURIComponent(dogName)}/form`).then(r => r.json())
    ])
    .then(([detailsData, formData]) => {
        displayDogDetails(detailsData, formData);
    })
    .catch(error => {
        console.error('Error loading dog details:', error);
        modalBody.innerHTML = '<div class="alert alert-danger">Error loading dog details.</div>';
    });
}

function displayDogDetails(detailsData, formData) {
    const modalBody = document.getElementById('dog-details-modal-body');
    
    if (!detailsData.success) {
        modalBody.innerHTML = '<div class="alert alert-danger">Error loading dog details.</div>';
        return;
    }
    
    const dog = detailsData.dog;
    const stats = detailsData.statistics;
    const venues = detailsData.venue_stats || [];
    const distances = detailsData.distance_stats || [];
    const form = formData.success ? formData.form_guide : [];
    
    const winRate = dog.total_races > 0 ? ((dog.total_wins / dog.total_races) * 100).toFixed(1) : '0.0';
    const placeRate = dog.total_races > 0 ? ((dog.total_places / dog.total_races) * 100).toFixed(1) : '0.0';
    
    let html = `
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Basic Statistics</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <small class="text-muted">Total Races:</small><br>
                                <strong>${dog.total_races}</strong>
                            </div>
                            <div class="col-6">
                                <small class="text-muted">Total Wins:</small><br>
                                <strong>${dog.total_wins}</strong>
                            </div>
                        </div>
                        <div class="row mt-2">
                            <div class="col-6">
                                <small class="text-muted">Win Rate:</small><br>
                                <span class="badge ${winRate > 20 ? 'bg-success' : winRate > 10 ? 'bg-warning' : 'bg-secondary'}">${winRate}%</span>
                            </div>
                            <div class="col-6">
                                <small class="text-muted">Place Rate:</small><br>
                                <span class="badge ${placeRate > 40 ? 'bg-success' : placeRate > 25 ? 'bg-warning' : 'bg-secondary'}">${placeRate}%</span>
                            </div>
                        </div>
                        ${dog.best_time ? `
                        <div class="mt-2">
                            <small class="text-muted">Best Time:</small><br>
                            <strong>${dog.best_time}s</strong>
                        </div>
                        ` : ''}
                        ${dog.average_position ? `
                        <div class="mt-2">
                            <small class="text-muted">Average Position:</small><br>
                            <strong>${dog.average_position}</strong>
                        </div>
                        ` : ''}
                        ${dog.last_race_date ? `
                        <div class="mt-2">
                            <small class="text-muted">Last Race:</small><br>
                            <small>${new Date(dog.last_race_date).toLocaleDateString()}</small>
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Performance by Venue</h6>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
    `;
    
    if (venues.length > 0) {
        venues.forEach(venue => {
            const venueWinRate = venue.races > 0 ? ((venue.wins / venue.races) * 100).toFixed(1) : '0.0';
            html += `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <strong>${venue.venue}</strong><br>
                        <small class="text-muted">${venue.races} races, ${venue.wins} wins</small>
                    </div>
                    <span class="badge ${venueWinRate > 20 ? 'bg-success' : venueWinRate > 10 ? 'bg-warning' : 'bg-secondary'}">${venueWinRate}%</span>
                </div>
            `;
        });
    } else {
        html += '<div class="text-muted">No venue statistics available.</div>';
    }
    
    html += `
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Recent Form Guide</h6>
                    </div>
                    <div class="card-body">
    `;
    
    if (form.length > 0) {
        html += `
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Venue</th>
                            <th>Position</th>
                            <th>Odds</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        form.forEach(race => {
            const positionBadge = race.position <= 3 ? 'bg-success' : race.position <= 6 ? 'bg-warning' : 'bg-secondary';
            html += `
                <tr>
                    <td>${new Date(race.race_date).toLocaleDateString()}</td>
                    <td>${race.venue}</td>
                    <td><span class="badge ${positionBadge}">${race.position}</span></td>
                    <td>${race.odds || 'N/A'}</td>
                    <td>${race.sectional_time || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        if (formData.form_trend) {
            const trendClass = formData.form_trend === 'improving' ? 'success' : 
                              formData.form_trend === 'declining' ? 'danger' : 'warning';
            html += `
                <div class="mt-2">
                    <small class="text-muted">Form Trend:</small>
                    <span class="badge bg-${trendClass}">${formData.form_trend}</span>
                </div>
            `;
        }
    } else {
        html += '<div class="text-muted">No recent form data available.</div>';
    }
    
    html += `
                    </div>
                </div>
            </div>
        </div>
    `;
    
    modalBody.innerHTML = html;
}

function loadAllDogs() {
    console.log('loadAllDogs called');
    const container = document.getElementById('all-dogs-container');
    const paginationContainer = document.getElementById('all-dogs-pagination');
    const loadBtn = document.getElementById('load-all-dogs-btn');
    
    console.log('Elements found:', {container, paginationContainer, loadBtn});
    
    if (!container || !paginationContainer || !loadBtn) {
        console.error('Required DOM elements not found!');
        return;
    }
    
    // Update button state
    loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    loadBtn.disabled = true;
    
    container.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading all dogs...</div>';
    
    const url = `/api/dogs/all?page=${currentPage}&per_page=5&sort_by=${currentSortBy}&order=${currentOrder}`;
    console.log('Fetching URL:', url);
    
    fetch(url)
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API Response:', data);
            if (data.success) {
                console.log('Dogs received:', data.dogs.length);
                displayAllDogs(data.dogs, data.pagination);
                displayPagination(data.pagination);
            } else {
                console.error('API returned error:', data.message);
                container.innerHTML = '<div class="alert alert-danger">Error loading dogs: ' + (data.message || 'Unknown error') + '</div>';
                paginationContainer.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error loading all dogs:', error);
            container.innerHTML = '<div class="alert alert-danger">Error loading dogs: ' + error.message + '</div>';
            paginationContainer.style.display = 'none';
        })
        .finally(() => {
            // Reset button state
            loadBtn.innerHTML = '<i class="fas fa-list"></i> Load All Dogs';
            loadBtn.disabled = false;
        });
}

function displayAllDogs(dogs, pagination) {
    const container = document.getElementById('all-dogs-container');
    
    if (!dogs || dogs.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No dogs found.</div>';
        return;
    }
    
    let html = `
        <div class="mb-3">
            <small class="text-muted">Showing ${dogs.length} of ${pagination.total_count} dogs (Page ${pagination.page} of ${pagination.total_pages})</small>
        </div>
        <div class="table-responsive">
            <table class="table table-hover">
                <thead class="table-light">
                    <tr>
                        <th>Dog Name</th>
                        <th>Total Races</th>
                        <th>Wins</th>
                        <th>Places</th>
                        <th>Win Rate</th>
                        <th>Place Rate</th>
                        <th>Avg Position</th>
                        <th>Best Time</th>
                        <th>Last Race</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    dogs.forEach(dog => {
        const winRate = dog.win_percentage || 0;
        const placeRate = dog.place_percentage || 0;
        const avgPosition = dog.average_position || 'N/A';
        const bestTime = dog.best_time || 'N/A';
        const lastRace = dog.last_race_date ? new Date(dog.last_race_date).toLocaleDateString() : 'N/A';
        
        html += `
            <tr>
                <td><strong>${dog.dog_name}</strong></td>
                <td>${dog.total_races}</td>
                <td>${dog.total_wins}</td>
                <td>${dog.total_places}</td>
                <td>
                    <span class="badge ${winRate > 20 ? 'bg-success' : winRate > 10 ? 'bg-warning' : 'bg-secondary'}">
                        ${winRate.toFixed(1)}%
                    </span>
                </td>
                <td>
                    <span class="badge ${placeRate > 40 ? 'bg-success' : placeRate > 25 ? 'bg-warning' : 'bg-secondary'}">
                        ${placeRate.toFixed(1)}%
                    </span>
                </td>
                <td>${avgPosition !== 'N/A' ? avgPosition.toFixed(1) : 'N/A'}</td>
                <td>${bestTime}${bestTime !== 'N/A' && typeof bestTime === 'number' ? 's' : ''}</td>
                <td><small>${lastRace}</small></td>
                <td>
                    <button class="btn btn-primary btn-sm" onclick="viewDogDetails('${dog.dog_name}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = html;
}

function displayPagination(pagination) {
    const paginationContainer = document.getElementById('all-dogs-pagination');
    
    if (pagination.total_pages <= 1) {
        paginationContainer.style.display = 'none';
        return;
    }
    
    let html = '<nav aria-label="Dogs pagination"><ul class="pagination pagination-sm justify-content-center mb-0">';
    
    // Previous button
    if (pagination.has_prev) {
        html += `
            <li class="page-item">
                <a class="page-link" href="#" onclick="changePage(${pagination.page - 1}); return false;">
                    <i class="fas fa-chevron-left"></i> Previous
                </a>
            </li>
        `;
    } else {
        html += '<li class="page-item disabled"><span class="page-link"><i class="fas fa-chevron-left"></i> Previous</span></li>';
    }
    
    // Page numbers (show up to 5 pages around current)
    const startPage = Math.max(1, pagination.page - 2);
    const endPage = Math.min(pagination.total_pages, pagination.page + 2);
    
    if (startPage > 1) {
        html += '<li class="page-item"><a class="page-link" href="#" onclick="changePage(1); return false;">1</a></li>';
        if (startPage > 2) {
            html += '<li class="page-item disabled"><span class="page-link">...</span></li>';
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        if (i === pagination.page) {
            html += `<li class="page-item active"><span class="page-link">${i}</span></li>`;
        } else {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(${i}); return false;">${i}</a></li>`;
        }
    }
    
    if (endPage < pagination.total_pages) {
        if (endPage < pagination.total_pages - 1) {
            html += '<li class="page-item disabled"><span class="page-link">...</span></li>';
        }
        html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(${pagination.total_pages}); return false;">${pagination.total_pages}</a></li>`;
    }
    
    // Next button
    if (pagination.has_next) {
        html += `
            <li class="page-item">
                <a class="page-link" href="#" onclick="changePage(${pagination.page + 1}); return false;">
                    Next <i class="fas fa-chevron-right"></i>
                </a>
            </li>
        `;
    } else {
        html += '<li class="page-item disabled"><span class="page-link">Next <i class="fas fa-chevron-right"></i></span></li>';
    }
    
    html += '</ul></nav>';
    
    paginationContainer.innerHTML = html;
    paginationContainer.style.display = 'block';
}

function changePage(page) {
    currentPage = page;
    loadAllDogs();
    
    // Scroll to top of the all dogs section
    document.getElementById('all-dogs-container').scrollIntoView({ behavior: 'smooth' });
}
