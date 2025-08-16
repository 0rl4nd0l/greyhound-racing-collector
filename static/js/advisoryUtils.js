/**
 * Advisory Rendering Utility
 * ==========================
 * 
 * Utility functions for rendering advisory messages with color-coded banners,
 * collapsible detail lists, tooltip help icons, and Bootstrap-compatible styling.
 */

// Initialize advisory styles
function initAdvisoryStyles() {
    if (!document.getElementById('advisory-utils-styles')) {
        const style = document.createElement('style');
        style.id = 'advisory-utils-styles';
        style.textContent = `
            .advisory-banner {
                border-left: 4px solid;
                border-radius: 0.375rem;
                margin-bottom: 1rem;
                position: relative;
            }
            
            .advisory-banner.alert-info {
                border-left-color: #0dcaf0;
            }
            
            .advisory-banner.alert-success {
                border-left-color: #198754;
            }
            
            .advisory-banner.alert-warning {
                border-left-color: #fd7e14;
            }
            
            .advisory-banner.alert-danger {
                border-left-color: #dc3545;
            }
            
            .advisory-banner.alert-secondary {
                border-left-color: #6c757d;
            }
            
            .advisory-icon {
                font-size: 1.25rem;
                margin-right: 0.75rem;
                opacity: 0.8;
            }
            
            .advisory-title {
                margin-bottom: 0.25rem;
                font-weight: 600;
            }
            
            .advisory-message {
                margin-bottom: 0.5rem;
                line-height: 1.5;
            }
            
            .advisory-details-toggle {
                font-size: 0.875rem;
                padding: 0.25rem 0.5rem;
                text-decoration: none;
                border: 1px solid transparent;
                border-radius: 0.25rem;
                transition: all 0.15s ease-in-out;
            }
            
            .advisory-details-toggle:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
            
            .advisory-details {
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid rgba(0, 0, 0, 0.125);
            }
            
            .advisory-help-icon {
                cursor: help;
                font-size: 0.875rem;
                opacity: 0.7;
                transition: opacity 0.15s ease-in-out;
            }
            
            .advisory-help-icon:hover {
                opacity: 1;
            }
            
            .advisory-loading {
                text-align: center;
                padding: 2rem;
            }
            
            .advisory-loading-text {
                margin-top: 0.75rem;
                color: #6c757d;
                font-size: 0.875rem;
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Map message types to Bootstrap-compatible class strings
 * @param {string} msgType - Message type (info, success, warning, danger)
 * @returns {string} Bootstrap alert class string
 */
function mapMessageTypeToClass(msgType) {
    const classes = {
        info: 'alert alert-info',
        success: 'alert alert-success',
        warning: 'alert alert-warning',
        danger: 'alert alert-danger',
        error: 'alert alert-danger' // Alias for danger
    };
    return classes[msgType] || 'alert alert-secondary';
}

/**
 * Get icon for message type
 * @param {string} msgType - Message type
 * @returns {string} Icon HTML
 */
function getAdvisoryIcon(msgType) {
    const icons = {
        info: '<i class="fas fa-info-circle advisory-icon text-info"></i>',
        success: '<i class="fas fa-check-circle advisory-icon text-success"></i>',
        warning: '<i class="fas fa-exclamation-triangle advisory-icon text-warning"></i>',
        danger: '<i class="fas fa-times-circle advisory-icon text-danger"></i>',
        error: '<i class="fas fa-times-circle advisory-icon text-danger"></i>'
    };
    return icons[msgType] || '<i class="fas fa-bell advisory-icon text-secondary"></i>';
}

/**
 * Generate a unique ID for collapsible elements
 * @param {string} title - Advisory title
 * @returns {string} Unique ID
 */
function generateAdvisoryId(title) {
    const timestamp = Date.now();
    const cleanTitle = title.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    return `advisory-${cleanTitle}-${timestamp}`;
}

/**
 * Render advisory with color-coded banners, collapsible details, and tooltip help icons
 * @param {Object} report - Advisory report object
 * @param {string} report.title - Advisory title
 * @param {string} report.message - Advisory message
 * @param {string} report.type - Advisory type (info, success, warning, danger)
 * @param {Array} [report.details] - Optional array of detail items
 * @param {string} [report.helpText] - Optional help text for tooltip
 * @param {HTMLElement} container - Container element to render in
 * @param {Object} [options] - Optional rendering options
 * @param {boolean} [options.dismissible] - Whether alert is dismissible (default: true)
 * @param {boolean} [options.showIcon] - Whether to show type icon (default: true)
 * @param {string} [options.detailsLabel] - Label for details toggle (default: 'Show Details')
 */
function renderAdvisory(report, container, options = {}) {
    // Initialize styles if not already done
    initAdvisoryStyles();
    
    // Extract report properties
    const { title, message, details, type = 'info', helpText } = report;
    
    // Set default options
    const opts = {
        dismissible: true,
        showIcon: true,
        detailsLabel: 'Show Details',
        ...options
    };
    
    // Generate unique IDs
    const advisoryId = generateAdvisoryId(title);
    const detailsId = `${advisoryId}-details`;
    
    // Get appropriate classes and icon
    const alertClass = mapMessageTypeToClass(type);
    const icon = opts.showIcon ? getAdvisoryIcon(type) : '';
    
    // Build advisory HTML
    let advisoryHTML = `
        <div class="${alertClass} advisory-banner alert-dismissible fade show" role="alert" id="${advisoryId}">
            <div class="d-flex align-items-start">
                ${icon}
                <div class="flex-grow-1">
                    <div class="advisory-title d-flex align-items-center">
                        <span>${title}</span>
                        ${helpText ? `
                            <i class="fas fa-question-circle advisory-help-icon ms-2" 
                               data-bs-toggle="tooltip" 
                               data-bs-placement="top" 
                               title="${helpText}"></i>
                        ` : ''}
                    </div>
                    <div class="advisory-message">${message}</div>
    `;
    
    // Add collapsible details if provided
    if (details && Array.isArray(details) && details.length > 0) {
        advisoryHTML += `
                    <button class="btn btn-link advisory-details-toggle p-0" 
                            type="button" 
                            data-bs-toggle="collapse" 
                            data-bs-target="#${detailsId}" 
                            aria-expanded="false" 
                            aria-controls="${detailsId}">
                        <i class="fas fa-chevron-down me-1"></i> ${opts.detailsLabel}
                    </button>
                    <div class="collapse advisory-details" id="${detailsId}">
                        <ul class="list-group list-group-flush">
        `;
        
        details.forEach(detail => {
            advisoryHTML += `<li class="list-group-item px-0">${detail}</li>`;
        });
        
        advisoryHTML += `
                        </ul>
                    </div>
        `;
    }
    
    advisoryHTML += `
                </div>
    `;
    
    // Add close button if dismissible
    if (opts.dismissible) {
        advisoryHTML += `
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
    }
    
    advisoryHTML += `
            </div>
        </div>
    `;
    
    // Add to container
    container.insertAdjacentHTML('beforeend', advisoryHTML);
    
    // Initialize tooltips if help text is present and Bootstrap tooltips are available
    if (helpText && typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipEl = container.querySelector(`#${advisoryId} .advisory-help-icon`);
        if (tooltipEl) {
            new bootstrap.Tooltip(tooltipEl);
        }
    }
    
    // Update chevron icon on collapse toggle
    const toggleBtn = container.querySelector(`#${advisoryId} .advisory-details-toggle`);
    if (toggleBtn) {
        const detailsEl = container.querySelector(`#${detailsId}`);
        if (detailsEl) {
            detailsEl.addEventListener('show.bs.collapse', () => {
                const chevron = toggleBtn.querySelector('.fas');
                if (chevron) {
                    chevron.classList.remove('fa-chevron-down');
                    chevron.classList.add('fa-chevron-up');
                }
            });
            
            detailsEl.addEventListener('hide.bs.collapse', () => {
                const chevron = toggleBtn.querySelector('.fas');
                if (chevron) {
                    chevron.classList.remove('fa-chevron-up');
                    chevron.classList.add('fa-chevron-down');
                }
            });
        }
    }
    
    return advisoryId;
}

/**
 * Clear advisory from container
 * @param {HTMLElement} container - Container element to clear
 */
function clearAdvisory(container) {
    // Dispose of any Bootstrap components before clearing
    const tooltips = container.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        const bsTooltip = bootstrap.Tooltip.getInstance(tooltip);
        if (bsTooltip) {
            bsTooltip.dispose();
        }
    });
    
    const alerts = container.querySelectorAll('.alert');
    alerts.forEach(alert => {
        const bsAlert = bootstrap.Alert.getInstance(alert);
        if (bsAlert) {
            bsAlert.dispose();
        }
    });
    
    container.innerHTML = '';
}

/**
 * Show loading state for advisory container
 * @param {HTMLElement} container - Container element
 * @param {string} [loadingText] - Optional loading text
 */
function showAdvisoryLoading(container, loadingText = 'Loading advisory...') {
    initAdvisoryStyles();
    
    container.innerHTML = `
        <div class="advisory-loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="advisory-loading-text">${loadingText}</div>
        </div>
    `;
}

// Attach to window for global reuse
window.AdvisoryUtils = {
    renderAdvisory,
    clearAdvisory,
    showAdvisoryLoading,
    mapMessageTypeToClass,
    getAdvisoryIcon,
    initAdvisoryStyles
};

// Log initialization
console.log('🎯 Advisory utilities initialized successfully');

