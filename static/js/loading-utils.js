/**
 * Frontend Loading & Error State Management
 * ==========================================
 * 
 * Utility functions for handling loading states, error display, 
 * token usage tracking, and input validation across the application.
 */

// Global loading state manager
class LoadingStateManager {
    constructor() {
        this.loadingStates = new Map();
        this.tokenUsage = {
            total_tokens: 0,
            total_cost: 0,
            sessions: []
        };
        this.initStyles();
    }

    initStyles() {
        // Inject loading spinner styles if not already present
        if (!document.getElementById('loading-utils-styles')) {
            const style = document.createElement('style');
            style.id = 'loading-utils-styles';
            style.textContent = `
                .btn-loading {
                    position: relative;
                    pointer-events: none;
                }
                
                .btn-loading .btn-text {
                    opacity: 0.6;
                }
                
                .btn-loading .loading-spinner {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 1rem;
                    height: 1rem;
                }
                
                .loading-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(255, 255, 255, 0.8);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                
                .loading-container {
                    position: relative;
                }
                
                .cost-badge {
                    font-family: 'Courier New', monospace;
                    font-size: 0.8rem;
                    margin-left: 0.5rem;
                }
                
                .token-usage-display {
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 0.375rem;
                    padding: 0.5rem;
                    margin-top: 0.5rem;
                    font-size: 0.85rem;
                }
                
                .input-validation-error {
                    border-color: #dc3545 !important;
                    box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25) !important;
                }
                
                .validation-feedback {
                    width: 100%;
                    margin-top: 0.25rem;
                    font-size: 0.875rem;
                    color: #dc3545;
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Show loading state for a button or container
     * @param {string|HTMLElement} elementOrId - Element or element ID
     * @param {string} loadingText - Optional loading text
     */
    showLoading(elementOrId, loadingText = 'Loading...') {
        const element = typeof elementOrId === 'string' 
            ? document.getElementById(elementOrId) 
            : elementOrId;
        
        if (!element) {
            console.warn(`Element not found: ${elementOrId}`);
            return;
        }

        const elementId = element.id || `loading-${Date.now()}`;
        
        if (element.tagName === 'BUTTON') {
            this.showButtonLoading(element, loadingText);
        } else {
            this.showContainerLoading(element, loadingText);
        }
        
        this.loadingStates.set(elementId, true);
    }

    /**
     * Hide loading state
     * @param {string|HTMLElement} elementOrId - Element or element ID
     */
    hideLoading(elementOrId) {
        const element = typeof elementOrId === 'string' 
            ? document.getElementById(elementOrId) 
            : elementOrId;
        
        if (!element) return;

        const elementId = element.id || '';
        
        if (element.tagName === 'BUTTON') {
            this.hideButtonLoading(element);
        } else {
            this.hideContainerLoading(element);
        }
        
        this.loadingStates.delete(elementId);
    }

    showButtonLoading(button, loadingText) {
        button.disabled = true;
        button.classList.add('btn-loading');
        
        // Store original content
        if (!button.dataset.originalContent) {
            button.dataset.originalContent = button.innerHTML;
        }
        
        // Add spinner
        const spinner = `<span class="spinner-border loading-spinner" role="status" aria-hidden="true"></span>`;
        const text = `<span class="btn-text">${loadingText}</span>`;
        button.innerHTML = spinner + text;
    }

    hideButtonLoading(button) {
        button.disabled = false;
        button.classList.remove('btn-loading');
        
        // Restore original content
        if (button.dataset.originalContent) {
            button.innerHTML = button.dataset.originalContent;
            delete button.dataset.originalContent;
        }
    }

    showContainerLoading(container, loadingText) {
        // Make container relative if not already
        if (getComputedStyle(container).position === 'static') {
            container.classList.add('loading-container');
        }
        
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-2">${loadingText}</div>
            </div>
        `;
        
        container.appendChild(overlay);
    }

    hideContainerLoading(container) {
        const overlay = container.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
        container.classList.remove('loading-container');
    }

    /**
     * Check if element is currently loading
     * @param {string|HTMLElement} elementOrId 
     * @returns {boolean}
     */
    isLoading(elementOrId) {
        const element = typeof elementOrId === 'string' 
            ? document.getElementById(elementOrId) 
            : elementOrId;
        
        const elementId = element?.id || '';
        return this.loadingStates.has(elementId);
    }
}

// Error display utilities
class ErrorDisplayManager {
    constructor() {
        this.alertContainer = this.createAlertContainer();
    }

    createAlertContainer() {
        let container = document.getElementById('alert-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'alert-container';
            container.className = 'position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1055';
            document.body.appendChild(container);
        }
        return container;
    }

    /**
     * Show dismissible Bootstrap alert
     * @param {string} message - Error message
     * @param {string} type - Alert type (danger, warning, info, success)
     * @param {number} duration - Auto-dismiss duration in ms (0 = no auto-dismiss)
     */
    showAlert(message, type = 'danger', duration = 5000) {
        const alertId = `alert-${Date.now()}`;
        const alert = document.createElement('div');
        alert.id = alertId;
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.setAttribute('role', 'alert');
        
        alert.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="flex-grow-1">
                    ${this.getAlertIcon(type)}
                    ${message}
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        this.alertContainer.appendChild(alert);
        
        // Auto-dismiss after duration
        if (duration > 0) {
            setTimeout(() => {
                const alertElement = document.getElementById(alertId);
                if (alertElement) {
                    const bsAlert = new bootstrap.Alert(alertElement);
                    bsAlert.close();
                }
            }, duration);
        }
        
        return alertId;
    }

    getAlertIcon(type) {
        const icons = {
            danger: '<i class="fas fa-exclamation-circle me-2"></i>',
            warning: '<i class="fas fa-exclamation-triangle me-2"></i>',
            info: '<i class="fas fa-info-circle me-2"></i>',
            success: '<i class="fas fa-check-circle me-2"></i>'
        };
        return icons[type] || '';
    }

    /**
     * Show server error in dismissible alert
     * @param {Object} error - Error object from server
     */
    showServerError(error) {
        let message = 'An unexpected error occurred.';
        
        if (typeof error === 'string') {
            message = error;
        } else if (error.message) {
            message = error.message;
        } else if (error.error) {
            message = error.error;
        }
        
        return this.showAlert(`Server Error: ${message}`, 'danger', 8000);
    }

    /**
     * Show validation errors for form inputs
     * @param {Object} errors - Object with field names as keys and error messages as values
     */
    showValidationErrors(errors) {
        Object.keys(errors).forEach(fieldName => {
            const field = document.getElementById(fieldName) || document.querySelector(`[name="${fieldName}"]`);
            if (field) {
                this.showFieldError(field, errors[fieldName]);
            }
        });
    }

    showFieldError(field, message) {
        // Add error class
        field.classList.add('input-validation-error');
        
        // Remove existing error message
        const existingError = field.parentNode.querySelector('.validation-feedback');
        if (existingError) {
            existingError.remove();
        }
        
        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'validation-feedback';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
        
        // Remove error styling when user starts typing
        field.addEventListener('input', () => {
            this.clearFieldError(field);
        }, { once: true });
    }

    clearFieldError(field) {
        field.classList.remove('input-validation-error');
        const errorDiv = field.parentNode.querySelector('.validation-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    clearAllErrors() {
        // Clear validation errors
        const errorFields = document.querySelectorAll('.input-validation-error');
        errorFields.forEach(field => this.clearFieldError(field));
        
        // Clear alerts
        const alerts = this.alertContainer.querySelectorAll('.alert');
        alerts.forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }
}

// Token usage and cost tracking
class TokenUsageTracker {
    constructor() {
        this.usage = this.loadUsage();
    }

    loadUsage() {
        const stored = localStorage.getItem('gpt-token-usage');
        return stored ? JSON.parse(stored) : {
            total_tokens: 0,
            total_cost: 0,
            sessions: []
        };
    }

    saveUsage() {
        localStorage.setItem('gpt-token-usage', JSON.stringify(this.usage));
    }

    /**
     * Record token usage from API response
     * @param {number} tokens - Number of tokens used
     * @param {number} cost - Estimated cost
     * @param {string} operation - Operation type (e.g., 'enhance_race')
     */
    recordUsage(tokens, cost, operation = 'unknown') {
        const session = {
            timestamp: new Date().toISOString(),
            tokens,
            cost,
            operation
        };
        
        this.usage.total_tokens += tokens;
        this.usage.total_cost += cost;
        this.usage.sessions.push(session);
        
        // Keep only last 50 sessions
        if (this.usage.sessions.length > 50) {
            this.usage.sessions = this.usage.sessions.slice(-50);
        }
        
        this.saveUsage();
        return session;
    }

    /**
     * Display token usage badge after an operation
     * @param {HTMLElement} container - Container element to append badge to
     * @param {number} tokens - Tokens used in this operation
     * @param {number} cost - Cost of this operation
     */
    displayUsageBadge(container, tokens, cost) {
        // Remove existing badge
        const existingBadge = container.querySelector('.token-usage-display');
        if (existingBadge) {
            existingBadge.remove();
        }
        
        const badge = document.createElement('div');
        badge.className = 'token-usage-display';
        badge.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>
                    <i class="fas fa-coins me-1"></i>
                    Tokens: <strong>${tokens.toLocaleString()}</strong>
                </span>
                <span class="cost-badge badge bg-info">
                    ~$${cost.toFixed(4)}
                </span>
            </div>
            <div class="text-muted small mt-1">
                Total: ${this.usage.total_tokens.toLocaleString()} tokens (~$${this.usage.total_cost.toFixed(2)})
            </div>
        `;
        
        container.appendChild(badge);
    }

    getTotalUsage() {
        return {
            total_tokens: this.usage.total_tokens,
            total_cost: this.usage.total_cost,
            session_count: this.usage.sessions.length
        };
    }

    resetUsage() {
        this.usage = {
            total_tokens: 0,
            total_cost: 0,
            sessions: []
        };
        this.saveUsage();
    }
}

// Input validation utilities
class InputValidator {
    /**
     * Validate file path list
     * @param {string} input - Comma-separated file paths
     * @returns {Object} - {valid: boolean, errors: string[], paths: string[]}
     */
    validateFilePathList(input) {
        const result = {
            valid: true,
            errors: [],
            paths: []
        };
        
        if (!input || typeof input !== 'string') {
            result.valid = false;
            result.errors.push('File path list is required');
            return result;
        }
        
        const paths = input.split(',')
            .map(path => path.trim())
            .filter(path => path.length > 0);
        
        if (paths.length === 0) {
            result.valid = false;
            result.errors.push('At least one file path is required');
            return result;
        }
        
        // Validate each path
        paths.forEach((path, index) => {
            if (!this.isValidFilePath(path)) {
                result.valid = false;
                result.errors.push(`Invalid file path at position ${index + 1}: "${path}"`);
            }
        });
        
        result.paths = paths;
        return result;
    }

    /**
     * Validate maximum races parameter
     * @param {string|number} input - Max races value
     * @returns {Object} - {valid: boolean, errors: string[], value: number}
     */
    validateMaxRaces(input) {
        const result = {
            valid: true,
            errors: [],
            value: 0
        };
        
        const num = parseInt(input, 10);
        
        if (isNaN(num)) {
            result.valid = false;
            result.errors.push('Max races must be a number');
            return result;
        }
        
        if (num < 1) {
            result.valid = false;
            result.errors.push('Max races must be at least 1');
            return result;
        }
        
        if (num > 100) {
            result.valid = false;
            result.errors.push('Max races cannot exceed 100');
            return result;
        }
        
        result.value = num;
        return result;
    }

    /**
     * Basic file path validation
     * @param {string} path - File path to validate
     * @returns {boolean}
     */
    isValidFilePath(path) {
        if (!path || typeof path !== 'string') return false;
        
        // Check for basic file path structure
        const pathRegex = /^[a-zA-Z0-9_\-\/\\\.]+\.(csv|json|txt)$/i;
        return pathRegex.test(path) && path.length <= 255;
    }

    /**
     * Validate form before submission
     * @param {HTMLFormElement} form - Form element to validate
     * @returns {Object} - {valid: boolean, errors: Object}
     */
    validateForm(form) {
        const result = {
            valid: true,
            errors: {}
        };
        
        const formData = new FormData(form);
        
        // Get validation rules from form
        const requiredFields = form.querySelectorAll('[required]');
        
        requiredFields.forEach(field => {
            const value = formData.get(field.name) || field.value;
            
            if (!value || (typeof value === 'string' && value.trim() === '')) {
                result.valid = false;
                result.errors[field.name] = `${field.name.replace(/[-_]/g, ' ')} is required`;
            }
        });
        
        // Custom validation based on field names/types
        const filePathFields = form.querySelectorAll('[data-validate="file-paths"]');
        filePathFields.forEach(field => {
            const validation = this.validateFilePathList(field.value);
            if (!validation.valid) {
                result.valid = false;
                result.errors[field.name] = validation.errors.join(', ');
            }
        });
        
        const maxRacesFields = form.querySelectorAll('[data-validate="max-races"]');
        maxRacesFields.forEach(field => {
            const validation = this.validateMaxRaces(field.value);
            if (!validation.valid) {
                result.valid = false;
                result.errors[field.name] = validation.errors.join(', ');
            }
        });
        
        return result;
    }
}

// Initialize global instances
const loadingManager = new LoadingStateManager();
const errorManager = new ErrorDisplayManager();
const tokenTracker = new TokenUsageTracker();
const validator = new InputValidator();

// Export utility functions
window.showLoading = (elementOrId, text) => loadingManager.showLoading(elementOrId, text);
window.hideLoading = (elementOrId) => loadingManager.hideLoading(elementOrId);
window.isLoading = (elementOrId) => loadingManager.isLoading(elementOrId);

window.showError = (message, type = 'danger', duration = 5000) => errorManager.showAlert(message, type, duration);
window.showServerError = (error) => errorManager.showServerError(error);
window.showValidationErrors = (errors) => errorManager.showValidationErrors(errors);
window.clearErrors = () => errorManager.clearAllErrors();

window.recordTokenUsage = (tokens, cost, operation) => tokenTracker.recordUsage(tokens, cost, operation);
window.displayTokenUsage = (container, tokens, cost) => tokenTracker.displayUsageBadge(container, tokens, cost);
window.getTokenUsage = () => tokenTracker.getTotalUsage();

window.validateFilePathList = (input) => validator.validateFilePathList(input);
window.validateMaxRaces = (input) => validator.validateMaxRaces(input);
window.validateForm = (form) => validator.validateForm(form);

// Export managers for advanced usage
window.LoadingUtils = {
    loadingManager,
    errorManager,
    tokenTracker,
    validator
};

console.log('🚀 Loading utilities initialized successfully');
