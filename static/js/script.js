// Greyhound Racing Dashboard JavaScript
// =====================================


document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initializeDashboard();
    
    // Auto-refresh stats every 30 seconds
    setInterval(refreshStats, 30000);
});

function initializeDashboard() {
    console.log('🐾 Greyhound Racing Dashboard initialized');
    
    // Add any initialization code here
    highlightActiveNavItem();
    initializeRefreshButtons();
}

function highlightActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');
    
    // Use Array.from to ensure forEach is available on all browsers
    Array.from(navLinks).forEach(link => {
        const href = link.getAttribute('href');
        // Clear any legacy inline background to maintain WCAG contrast
        if (link.style && link.style.backgroundColor) {
            link.style.backgroundColor = '';
        }
        if (href === currentPath) {
            // Prefer semantic active state and accessible name
            link.classList.add('active');
            if (!link.getAttribute('aria-current')) {
                link.setAttribute('aria-current', 'page');
            }
            // Ensure high-contrast text color is enforced via CSS, not inline styles
        }
    });
}

function initializeRefreshButtons() {
    const refreshButtons = document.querySelectorAll('[data-refresh]');
    
    // Use Array.from to ensure forEach is available on all browsers
    Array.from(refreshButtons).forEach(button => {
        button.addEventListener('click', function() {
            const target = this.getAttribute('data-refresh');
            refreshSection(target);
        });
    });
}

function refreshStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            console.log('📊 Stats refreshed:', data);
            // Update stats on the page if needed
        })
        .catch(error => {
            console.error('Error refreshing stats:', error);
        });
}

function refreshSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.style.opacity = '0.5';
        
        // Simulate refresh - in reality, this would update the content
        setTimeout(() => {
            section.style.opacity = '1';
        }, 500);
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// API interaction functions
async function fetchRecentRaces(limit = 10) {
    try {
        const response = await fetch(`/api/recent_races?limit=${limit}`);
        const data = await response.json();
        const racesArray = Array.isArray(data.races) ? data.races : Object.values(data.races || {});
        return racesArray;
    } catch (error) {
        console.error('Error fetching recent races:', error);
        return [];
    }
}

async function fetchRaceDetails(raceId) {
    try {
        const response = await fetch(`/api/race/${raceId}`);
        const data = await response.json();
        return data.race_data;
    } catch (error) {
        console.error('Error fetching race details:', error);
        return null;
    }
}

// Export functions for global use
window.dashboardUtils = {
    refreshStats,
    refreshSection,
    formatFileSize,
    formatDate,
    showNotification,
    fetchRecentRaces,
    fetchRaceDetails
};
