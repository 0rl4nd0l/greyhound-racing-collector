// Model Registry Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeModelRegistry();
    
    // Setup event listeners
    document.getElementById('start-training-btn').addEventListener('click', startModelTraining);
    
    // Refresh data every 30 seconds
    setInterval(refreshModelRegistry, 30000);
});

function initializeModelRegistry() {
    refreshModelRegistry();
}

function refreshModelRegistry() {
    // Load registry status
    fetch('/api/model/registry/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateBestModels(data.best_models);
                updateRegistryStatus(data);
                updateModelsTable(data.all_models);
            }
        })
        .catch(error => {
            console.error('Error fetching registry status:', error);
            showNotification('Error loading model registry data', 'error');
        });
}

function updateBestModels(bestModels) {
    const container = document.getElementById('best-models-container');
    let html = '';
    
    for (const [predType, model] of Object.entries(bestModels)) {
        html += `
            <div class="mb-3">
                <h6>${formatPredictionType(predType)}</h6>
                <div class="card bg-light">
                    <div class="card-body p-2">
                        <small class="text-muted">Model ID:</small> ${model.model_id}<br>
                        <small class="text-muted">Version:</small> ${model.version}<br>
                        <small class="text-muted">Score:</small> ${formatScore(model.performance_score)}<br>
                        <small class="text-muted">Created:</small> ${formatDate(model.created_at)}
                    </div>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function updateRegistryStatus(data) {
    const container = document.getElementById('registry-status-container');
    
    const html = `
        <div class="d-flex justify-content-between mb-3">
            <div>
                <h6 class="mb-2">Registry Status</h6>
                <span class="badge bg-success">Active</span>
            </div>
            <div class="text-end">
                <button class="btn btn-primary btn-sm" onclick="showTrainingModal()">
                    Train New Model
                </button>
            </div>
        </div>
        <div class="row g-2">
            <div class="col-6">
                <div class="border rounded p-2">
                    <small class="text-muted d-block">Total Models</small>
                    <strong>${data.total_models}</strong>
                </div>
            </div>
            <div class="col-6">
                <div class="border rounded p-2">
                    <small class="text-muted d-block">Active Models</small>
                    <strong>${Object.keys(data.best_models).length}</strong>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function updateModelsTable(models) {
    const tbody = document.getElementById('models-table-body');
    tbody.innerHTML = '';
    
    models.forEach(model => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${model.model_id}</td>
            <td>${formatPredictionType(model.prediction_type)}</td>
            <td>v${model.version}</td>
            <td>${formatScore(model.performance_score)}</td>
            <td>${formatDate(model.created_at)}</td>
            <td>
                <span class="badge bg-${model.is_active ? 'success' : 'secondary'}">
                    ${model.is_active ? 'Active' : 'Inactive'}
                </span>
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="showModelDetails('${model.model_id}')">
                        Details
                    </button>
                    <button class="btn btn-outline-secondary" onclick="downloadModel('${model.model_id}')">
                        Download
                    </button>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function showTrainingModal() {
    const modal = new bootstrap.Modal(document.getElementById('trainingModal'));
    modal.show();
}

function startModelTraining() {
    const form = document.getElementById('training-form');
    const formData = new FormData(form);
    
    const data = {
        prediction_type: formData.get('prediction_type'),
        training_data_days: parseInt(formData.get('training_data_days')),
        force_retrain: formData.get('force_retrain') === 'on'
    };
    
    fetch('/api/model/training/trigger', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Model training initiated successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('trainingModal')).hide();
            // Refresh after a short delay to show new model
            setTimeout(refreshModelRegistry, 2000);
        } else {
            showNotification(`Training error: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error starting training:', error);
        showNotification('Error starting model training', 'error');
    });
}

async function showModelDetails(modelId) {
    try {
        const [performanceResponse, monitoringResponse] = await Promise.all([
            fetch(`/api/model/performance?model_id=${modelId}`),
            fetch(`/api/model/monitoring/drift`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_id: modelId })
            })
        ]);
        
        const performanceData = await performanceResponse.json();
        const monitoringData = await monitoringResponse.json();
        
        // Show details in a modal or detailed view
        // Implementation depends on your UI requirements
    } catch (error) {
        console.error('Error fetching model details:', error);
        showNotification('Error loading model details', 'error');
    }
}

function downloadModel(modelId) {
    window.location.href = `/api/model/download/${modelId}`;
}

// Utility functions
function formatPredictionType(type) {
    return type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function formatScore(score) {
    return (score * 100).toFixed(1) + '%';
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleString();
}

function showNotification(message, type = 'info') {
    // Use your notification system here
    console.log(`${type.toUpperCase()}: ${message}`);
}
