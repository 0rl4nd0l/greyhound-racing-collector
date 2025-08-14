#!/usr/bin/env python3
"""
Model Training API Blueprint
===========================

Flask blueprint for registry-aware model training API endpoints.
Provides endpoints for triggering training, checking status, and listing models.

Routes:
- POST /api/model/training/trigger - Start model training
- GET /api/model/registry/status - Get registry status and training jobs
- GET /api/model/list_trainable - List available trainable models

Author: AI Assistant
Date: January 2025
"""

import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Blueprint, request, jsonify
from model_registry import get_model_registry

# Create a Blueprint for model training API
model_training_bp = Blueprint('model_training_api', __name__)

# In-memory storage for training job statuses
training_jobs: Dict[str, Dict[str, Any]] = {}


@model_training_bp.route('/api/model/training/trigger', methods=['POST'])
def trigger_training():
    """Trigger model training with optional parameters.
    
    Body: {
        model_id?: str,
        prediction_type?: str,
        training_data_days?: int,
        force_retrain?: bool
    }
    
    Returns: { success: true, job_id }
    """
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id')
        prediction_type = data.get('prediction_type', 'win')
        training_data_days = data.get('training_data_days', 30)
        force_retrain = data.get('force_retrain', False)
        
        # If model_id is omitted, get best model for prediction type
        if not model_id:
            try:
                registry = get_model_registry()
                best_models = registry.get_best_models()
                model_id = best_models.get(prediction_type, {}).get('model_id')
                
                if not model_id:
                    # Default to comprehensive training if no specific model found
                    model_id = 'comprehensive_training'
            except Exception as e:
                model_id = 'comprehensive_training'  # Fallback
        
        # Generate unique job ID
        job_id = f"training_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Create training job entry
        training_jobs[job_id] = {
            'id': job_id,
            'status': 'starting',
            'progress': 0,
            'model_id': model_id,
            'prediction_type': prediction_type,
            'training_data_days': training_data_days,
            'force_retrain': force_retrain,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'error_message': None,
            'thread': None
        }
        
        # Start training in background thread
        thread = threading.Thread(
            target=retrain_worker, 
            args=(model_id, job_id, training_data_days, force_retrain),
            daemon=True
        )
        thread.start()
        
        training_jobs[job_id]['thread'] = thread
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True, 
            'job_id': job_id,
            'message': f'Training job {job_id} started for model {model_id}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to trigger training: {str(e)}'
        }), 500


@model_training_bp.route('/api/model/registry/status', methods=['GET'])
def get_registry_status():
    """Return full registry dump plus current training_jobs.
    
    Returns: {
        models: [...],
        training_jobs: { job_id: {id, progress, status} }
    }
    """
    try:
        # Get model registry status
        registry = get_model_registry()
        models = []
        
        try:
            raw_models = registry.list_models()
            for model in raw_models:
                if hasattr(model, '__dict__'):
                    # Convert model object to dictionary
                    model_dict = {
                        'model_id': getattr(model, 'model_id', 'unknown'),
                        'model_name': getattr(model, 'model_name', 'Unknown'),
                        'model_type': getattr(model, 'model_type', 'Unknown'),
                        'version': getattr(model, 'training_timestamp', 'Unknown'),
                        'prediction_type': getattr(model, 'model_type', 'win'),  # Infer from type
                        'created_at': getattr(model, 'training_timestamp', 'Unknown'),
                        'accuracy': getattr(model, 'accuracy', 0.0),
                        'auc': getattr(model, 'auc', 0.0),
                        'f1_score': getattr(model, 'f1_score', 0.0),
                        'is_active': getattr(model, 'is_active', False),
                        'is_best': getattr(model, 'is_best', False),
                        'training_samples': getattr(model, 'training_samples', 0),
                        'features_count': getattr(model, 'features_count', 0)
                    }
                    models.append(model_dict)
                elif isinstance(model, dict):
                    models.append(model)
        except Exception as e:
            models = [{'error': f'Failed to load models: {str(e)}'}]
        
        # Get current training jobs status
        jobs_status = {}
        for job_id, job in training_jobs.items():
            jobs_status[job_id] = {
                'id': job['id'],
                'status': job['status'],
                'progress': job['progress'],
                'model_id': job.get('model_id'),
                'prediction_type': job.get('prediction_type'),
                'created_at': job.get('created_at'),
                'started_at': job.get('started_at'),
                'completed_at': job.get('completed_at'),
                'error_message': job.get('error_message')
            }
        
        return jsonify({
            'success': True,
            'models': models,
            'training_jobs': jobs_status,
            'registry_info': {
                'total_models': len(models),
                'active_jobs': len([j for j in training_jobs.values() if j['status'] == 'running']),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get registry status: {str(e)}',
            'models': [],
            'training_jobs': {}
        }), 500


@model_training_bp.route('/api/model/list_trainable', methods=['GET'])
def list_trainable_models():
    """Return list of trainable models with metadata.
    
    Returns: {
        models: [{
            model_id: str,
            model_name: str,
            version: str,
            prediction_type: str,
            created_at: str
        }]
    }
    """
    try:
        registry = get_model_registry()
        trainable_models = []
        
        try:
            # Get all models from registry
            all_models = registry.list_models()
            
            # Filter for active/latest models (you can modify this logic)
            for model in all_models:
                if hasattr(model, '__dict__'):
                    model_dict = {
                        'model_id': getattr(model, 'model_id', f'model_{len(trainable_models)}'),
                        'model_name': getattr(model, 'model_name', 'Unknown Model'),
                        'version': getattr(model, 'training_timestamp', 'v1.0'),
                        'prediction_type': getattr(model, 'model_type', 'win'),
                        'created_at': getattr(model, 'training_timestamp', datetime.now().isoformat()),
                        'accuracy': getattr(model, 'accuracy', 0.0),
                        'is_active': getattr(model, 'is_active', False),
                        'training_samples': getattr(model, 'training_samples', 0)
                    }
                    trainable_models.append(model_dict)
                elif isinstance(model, dict):
                    # Handle dictionary models
                    trainable_models.append({
                        'model_id': model.get('model_id', f'model_{len(trainable_models)}'),
                        'model_name': model.get('model_name', 'Unknown Model'),
                        'version': model.get('version', 'v1.0'),
                        'prediction_type': model.get('prediction_type', 'win'),
                        'created_at': model.get('created_at', datetime.now().isoformat()),
                        'accuracy': model.get('accuracy', 0.0),
                        'is_active': model.get('is_active', False),
                        'training_samples': model.get('training_samples', 0)
                    })
            
            # Add default trainable model types if no models exist
            if not trainable_models:
                default_models = [
                    {
                        'model_id': 'comprehensive_training',
                        'model_name': 'Comprehensive ML Model',
                        'version': 'v1.0',
                        'prediction_type': 'win',
                        'created_at': datetime.now().isoformat(),
                        'accuracy': 0.0,
                        'is_active': True,
                        'training_samples': 0
                    },
                    {
                        'model_id': 'automated_training',
                        'model_name': 'Automated ML Training',
                        'version': 'v1.0',
                        'prediction_type': 'place',
                        'created_at': datetime.now().isoformat(),
                        'accuracy': 0.0,
                        'is_active': True,
                        'training_samples': 0
                    }
                ]
                trainable_models = default_models
                
        except Exception as e:
            # Fallback to default models if registry access fails
            trainable_models = [
                {
                    'model_id': 'comprehensive_training',
                    'model_name': 'Comprehensive ML Model',
                    'version': 'v1.0',
                    'prediction_type': 'win',
                    'created_at': datetime.now().isoformat(),
                    'accuracy': 0.0,
                    'is_active': True,
                    'training_samples': 0
                }
            ]
        
        return jsonify({
            'success': True,
            'models': trainable_models,
            'count': len(trainable_models)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to list trainable models: {str(e)}',
            'models': []
        }), 500


def retrain_worker(job_id: str, model_id: str, params: Dict[str, Any]):
    """Background worker function for model retraining.
    
    Updates training_jobs[job_id] with progress and status.
    Calls existing run_training_background or a light wrapper.
    """
    try:
        if job_id not in training_jobs:
            return
            
        job = training_jobs[job_id]
        
        # Update job status
        job['status'] = 'running'
        job['progress'] = 10
        
        # Import the existing training function
        try:
            # Try to import the existing run_training_background from the main app
            # We'll need to avoid circular imports
            import sys
            import os
            
            # Add current directory to path to import app functions
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            
            job['progress'] = 30
            
            # Call the appropriate training function based on model_id
            if model_id == 'comprehensive_training':
                # Comprehensive training logic
                job['progress'] = 50
                try:
                    from retrain_ml_models import retrain_models, extract_training_data, prepare_training_data
                    # Load metadata and data slice
                    metadata = registry.get_model_metadata(model_id)
                    training_samples = extract_training_data()
                    X, y, feature_columns = prepare_training_data(training_samples, metadata)
                    job['progress'] = 70
                    results = retrain_models(X, y, feature_columns)
                    job['progress'] = 90
                    if results:
                        job['status'] = 'completed'
                        job['progress'] = 100
                        job['completed_at'] = datetime.now().isoformat()
                        # Register new model
                        registry.register_model(model_id, metadata)
                    else:
                        job['status'] = 'failed'
                        job['error_message'] = 'Training completed but no results returned'
                except Exception as training_error:
                    job['status'] = 'failed'
                    job['error_message'] = f'Training error: {str(training_error)}'
            else:
                # Custom model training - call existing run_training_background
                job['progress'] = 50
                
                try:
                    # Import existing training function carefully to avoid circular imports
                    from app import run_training_background
                    
                    job['progress'] = 70
                    
                    # Call existing training function
                    result = run_training_background(model_id)
                    
                    job['progress'] = 90
                    
                    # Check global training status from app.py
                    from app import training_status as global_training_status
                    
                    if global_training_status.get('completed', False):
                        job['status'] = 'completed'
                        job['progress'] = 100
                        job['completed_at'] = datetime.now().isoformat()
                    elif global_training_status.get('error'):
                        job['status'] = 'failed'
                        job['error_message'] = global_training_status.get('error')
                    else:
                        job['status'] = 'completed'  # Assume success if no error
                        job['progress'] = 100
                        job['completed_at'] = datetime.now().isoformat()
                        
                except Exception as training_error:
                    job['status'] = 'failed'
                    job['error_message'] = f'Custom training error: {str(training_error)}'
                    
        except ImportError as e:
            job['status'] = 'failed'
            job['error_message'] = f'Failed to import training modules: {str(e)}'
            
    except Exception as e:
        if job_id in training_jobs:
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error_message'] = f'Worker error: {str(e)}'
        
    finally:
        # Clean up thread reference
        if job_id in training_jobs and 'thread' in training_jobs[job_id]:
            training_jobs[job_id]['thread'] = None

