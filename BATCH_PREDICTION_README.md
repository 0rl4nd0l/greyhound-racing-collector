# Robust Batch Prediction Workflow

## Overview

The Robust Batch Prediction Workflow provides comprehensive batch processing capabilities for greyhound racing predictions. It includes memory-efficient pagination, streaming support, progress callbacks, and extensive error handling for production use.

## Features

### ✅ Core Functionality
- **Batch Processing**: Process multiple CSV files concurrently
- **Progress Tracking**: Real-time progress callbacks and monitoring
- **Error Recovery**: Robust error handling with detailed reporting
- **Memory Management**: Pagination and streaming to prevent memory exhaustion
- **Concurrent Execution**: Multi-threaded processing with configurable workers

### ✅ Edge Case Handling
- **Empty Files**: Graceful handling of empty CSV files and rows
- **Unicode Support**: Full support for international characters (UTF-8, Latin-1)
- **Mixed Formats**: Handles varied track conditions and data formats
- **Large Files**: Memory-efficient processing of large CSV uploads
- **Malformed Data**: Resilient parsing with error recovery

### ✅ CLI Interface
- **Command Line Tool**: `cli_batch_predictor.py` for terminal usage
- **Job Management**: Create, monitor, and cancel batch jobs
- **Progress Monitoring**: Real-time progress bars and status updates
- **Multiple Modes**: Single file, batch directory, or upcoming races

### ✅ API Endpoints
- **REST API**: Integration with Flask web application
- **Progress Callbacks**: JSONP support for cross-origin requests
- **Streaming API**: Server-sent events for real-time updates
- **Job Management**: Create, monitor, cancel, and list jobs

## Installation

### Prerequisites
```bash
pip install pandas psutil pytest
```

### Files Required
- `batch_prediction_pipeline.py` - Core batch processing engine
- `cli_batch_predictor.py` - Command-line interface
- `test_batch_prediction_edge_cases.py` - Comprehensive test suite
- `run_batch_tests.py` - Test runner utility

## Usage

### 1. Command Line Interface

#### Basic Usage
```bash
# Predict a single race file
python cli_batch_predictor.py --file race.csv

# Predict all files in a directory
python cli_batch_predictor.py --batch ./upcoming_races/

# Predict all upcoming races
python cli_batch_predictor.py --upcoming-races

# Check job status
python cli_batch_predictor.py --job-status abc123

# List all jobs
python cli_batch_predictor.py --list-jobs
```

#### Advanced Options
```bash
# Quiet mode (minimal output)
python cli_batch_predictor.py --file race.csv --quiet

# Enable detailed progress callbacks
python cli_batch_predictor.py --batch ./races/ --progress-callback
```

### 2. Python API

#### Basic Batch Processing
```python
from batch_prediction_pipeline import BatchPredictionPipeline

# Initialize pipeline
pipeline = BatchPredictionPipeline()

# Create batch job
job_id = pipeline.create_batch_job(
    name="My Batch Job",
    input_files=["race1.csv", "race2.csv", "race3.csv"],
    output_dir="./predictions",
    batch_size=10,
    max_workers=3
)

# Run job
pipeline.run_batch_job(job_id)

# Check status
job = pipeline.get_job_status(job_id)
print(f"Status: {job.status}, Progress: {job.progress}%")
```

#### With Progress Callbacks
```python
def progress_callback(job_status):
    print(f"Progress: {job_status.progress}% - {job_status.completed_files}/{job_status.total_files}")

job_id = pipeline.create_batch_job(
    name="Monitored Job",
    input_files=csv_files,
    output_dir="./output",
    progress_callback=progress_callback
)
```

### 3. Flask API Endpoints

#### Start Batch Job
```bash
curl -X POST http://localhost:5002/api/batch/predict \
  -H "Content-Type: application/json" \
  -d '{"files": ["race1.csv", "race2.csv"]}'
```

#### Check Job Status
```bash
curl http://localhost:5002/api/batch/status/abc123
```

#### Progress with Callback
```bash
curl "http://localhost:5002/api/batch/progress/abc123?callback=myCallback"
```

#### Streaming Updates
```bash
curl -X POST http://localhost:5002/api/batch/stream \
  -H "Content-Type: application/json" \
  -d '{"files": ["race1.csv"], "batch_size": 5}'
```

## Configuration

### Batch Job Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Human-readable job name | Required |
| `input_files` | List of CSV file paths | Required |
| `output_dir` | Directory for prediction outputs | Required |
| `batch_size` | Files processed per batch | 10 |
| `max_workers` | Maximum concurrent workers | 3 |
| `progress_callback` | Function for progress updates | None |

### Memory Management

The pipeline automatically manages memory usage through:
- **Pagination**: Processes files in configurable batches
- **Streaming**: Reads large files in chunks
- **Garbage Collection**: Automatic cleanup of processed data
- **Memory Monitoring**: Tracks and limits memory usage

## Testing

### Run All Tests
```bash
python run_batch_tests.py
```

### Run Specific Test Suites
```bash
# Edge cases only
python run_batch_tests.py --edge-cases

# Integration tests only
python run_batch_tests.py --integration

# Quick tests (no slow tests)
python run_batch_tests.py --quick

# Verbose output
python run_batch_tests.py --verbose
```

### Test Coverage

The test suite covers:
- ✅ Empty CSV files and rows
- ✅ Unicode characters (UTF-8, Cyrillic, Japanese)
- ✅ Mixed track conditions and formats
- ✅ Large file memory handling (10k+ rows)
- ✅ Pagination and streaming
- ✅ Progress callbacks
- ✅ Concurrent job execution
- ✅ Job cancellation
- ✅ Error recovery and reporting
- ✅ Malformed CSV handling
- ✅ Encoding detection
- ✅ CLI integration
- ✅ API endpoint integration

## Error Handling

### Common Issues and Solutions

#### Empty or Missing Files
```python
# Files are validated before processing
# Empty files are marked as failed but don't stop the batch
job = pipeline.get_job_status(job_id)
print(f"Failed files: {job.failed_files}")
print(f"Errors: {job.error_messages}")
```

#### Memory Issues with Large Files
```python
# Reduce batch size and workers for large files
job_id = pipeline.create_batch_job(
    name="Large Files",
    input_files=large_csv_files,
    output_dir="./output",
    batch_size=2,  # Smaller batches
    max_workers=1  # Single worker
)
```

#### Unicode Encoding Problems
```python
# Pipeline automatically detects and handles encoding
# Supports UTF-8, Latin-1, and other common encodings
# Falls back gracefully if encoding detection fails
```

## Performance Optimization

### Best Practices

1. **Batch Size**: 
   - Small files: batch_size=20-50
   - Large files: batch_size=5-10
   - Very large files: batch_size=1-2

2. **Worker Threads**:
   - CPU-bound: max_workers = CPU cores
   - I/O-bound: max_workers = 2-4x CPU cores
   - Memory-limited: max_workers = 1-2

3. **Memory Management**:
   - Monitor memory usage with large datasets
   - Use streaming for files >100MB
   - Enable garbage collection between batches

### Performance Monitoring
```python
import psutil
import os

def monitor_resources():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    return f"Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%"

# Use in progress callback
def progress_with_monitoring(job_status):
    print(f"Progress: {job_status.progress}% - {monitor_resources()}")
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Batch Prediction Workflow               │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface          │  Flask API Endpoints              │
│  • cli_batch_predictor  │  • /api/batch/predict             │
│  • Progress monitoring  │  • /api/batch/status              │
│  • Job management       │  • /api/batch/progress            │
│                         │  • /api/batch/stream              │
├─────────────────────────────────────────────────────────────┤
│                   BatchPredictionPipeline                   │
│  • Job creation/management                                  │
│  • Progress tracking                                        │
│  • Worker thread pool                                       │
│  • Error handling                                           │
├─────────────────────────────────────────────────────────────┤
│  File Processing        │  Memory Management                │
│  • CSV parsing          │  • Pagination                     │
│  • Unicode handling     │  • Streaming                      │
│  • Format validation    │  • Garbage collection             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Validation**: Check file existence and format
2. **Job Creation**: Initialize job with metadata and settings
3. **Batch Processing**: Divide files into manageable batches
4. **Worker Threads**: Process batches concurrently
5. **Progress Tracking**: Update status and invoke callbacks
6. **Error Handling**: Capture and report individual failures
7. **Output Generation**: Save predictions to specified directory
8. **Cleanup**: Release resources and update final status

## Integration

### Web Application Integration
The batch prediction system integrates seamlessly with the Flask web application through:
- API endpoints for job management
- Progress callbacks for real-time updates
- WebSocket support for streaming progress
- Database integration for job persistence

### External System Integration
- **File System**: Automatic directory creation and file management
- **Database**: Optional job state persistence
- **Monitoring**: Integration with system monitoring tools
- **Logging**: Comprehensive error and activity logging

## Troubleshooting

### Common Issues

1. **Jobs Not Starting**
   - Check file permissions
   - Verify input files exist
   - Ensure output directory is writable

2. **Memory Errors**
   - Reduce batch_size
   - Decrease max_workers
   - Check available system memory

3. **Unicode Errors**
   - Files should be UTF-8 encoded
   - Pipeline auto-detects most encodings
   - Check for BOM (Byte Order Mark) issues

4. **Performance Issues**
   - Monitor CPU and memory usage
   - Adjust batch_size and max_workers
   - Check for I/O bottlenecks

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create pipeline with debug info
pipeline = BatchPredictionPipeline(debug=True)
```

## Support

For issues and questions:
1. Check the test suite for examples
2. Review error messages in job.error_messages
3. Enable debug logging for detailed information
4. Refer to the Flask API endpoint documentation

## License

This batch prediction workflow is part of the Greyhound Racing Collector project.
