import pytest
import os
from unittest.mock import MagicMock, patch
from tempfile import NamedTemporaryFile
from batch_prediction_pipeline import BatchPredictionPipeline, BatchProcessingConfig, BatchJob

@pytest.fixture
def mock_csv_file():
    def _create_csv_file(content):
        with NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as f:
            f.write(content)
            return f.name
    return _create_csv_file

@pytest.fixture
def batch_pipeline():
    config = BatchProcessingConfig(
        chunk_size=1,  # Small size for testing streaming
        max_memory_mb=512,
        timeout_seconds=300
    )
    return BatchPredictionPipeline(config)


def test_empty_row_handling(mock_csv_file, batch_pipeline):
    csv_content = """dog_name,track,odds
Luna,MEADOWS,5.0
,,
Oscar,WENTWORTH,3.0
"""
    csv_file = mock_csv_file(csv_content)

    job_id = batch_pipeline.create_batch_job(
        name="Test Job Empty Rows",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)

    assert job.status == 'completed'
    assert job.completed_files == 1


def test_unicode_handling(mock_csv_file, batch_pipeline):
    unicode_content = u"""dog_name,track,odds
Luna,MEADOWS,5.0
Félix,WENTWORTH,2.9
"""
    csv_file = mock_csv_file(unicode_content)

    job_id = batch_pipeline.create_batch_job(
        name="Test Job Unicode",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)

    assert job.status == 'completed'
    assert job.completed_files == 1


def test_mixed_tracks_handling(mock_csv_file, batch_pipeline):
    mixed_content = u"""dog_name,track,odds
Luna,MEADOWS,5.0
Max,THE MEADOWS,3.2
Oscar,WENTWORTH,2.5
"""
    csv_file = mock_csv_file(mixed_content)

    job_id = batch_pipeline.create_batch_job(
        name="Test Job Mixed Tracks",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)

    assert job.status == 'completed'
    assert job.completed_files == 1

def test_streaming_large_file(mock_csv_file, batch_pipeline):
    """Test streaming processing for large files"""
    # Create a large CSV with many rows to trigger streaming
    large_content = "dog_name,track,odds\n"
    for i in range(1500):  # More than default chunk size
        large_content += f"Dog{i},MEADOWS,{5.0 + i * 0.1}\n"
    
    csv_file = mock_csv_file(large_content)
    
    job_id = batch_pipeline.create_batch_job(
        name="Test Large File Streaming",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)
    
    # Should complete even with large file
    assert job.status in ['completed', 'completed_with_errors']
    assert job.completed_files == 1
    
    # Clean up
    os.unlink(csv_file)

def test_pagination_memory_management(mock_csv_file):
    """Test pagination and memory management"""
    # Create config with very small chunk size to force pagination
    config = BatchProcessingConfig(
        chunk_size=5,  # Very small chunks
        max_memory_mb=256,
        timeout_seconds=60
    )
    pipeline = BatchPredictionPipeline(config)
    
    # Create CSV with enough rows to require multiple chunks
    content = "dog_name,track,odds\n"
    for i in range(50):
        content += f"TestDog{i},MEADOWS,{3.0 + i * 0.1}\n"
    
    csv_file = mock_csv_file(content)
    
    job_id = pipeline.create_batch_job(
        name="Pagination Test",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = pipeline.run_batch_job(job_id)
    
    # Should handle chunked processing
    assert job.status in ['completed', 'completed_with_errors']
    
    # Clean up
    os.unlink(csv_file)

def test_invalid_csv_handling(mock_csv_file, batch_pipeline):
    """Test handling of invalid CSV files"""
    invalid_content = "This is not a valid CSV file\nNo proper structure here"
    csv_file = mock_csv_file(invalid_content)
    
    job_id = batch_pipeline.create_batch_job(
        name="Invalid CSV Test",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)
    
    # Should handle invalid files gracefully
    assert job.failed_files == 1
    assert len(job.error_messages) > 0
    
    # Clean up
    os.unlink(csv_file)

def test_job_cancellation(mock_csv_file, batch_pipeline):
    """Test job cancellation functionality"""
    content = "dog_name,track,odds\nLuna,MEADOWS,5.0\n"
    csv_file = mock_csv_file(content)
    
    job_id = batch_pipeline.create_batch_job(
        name="Cancellation Test",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    
    # Cancel the job before completion
    cancelled = batch_pipeline.cancel_job(job_id)
    assert cancelled == True
    
    job = batch_pipeline.get_job_status(job_id)
    assert job.status == "cancelled"
    
    # Clean up
    os.unlink(csv_file)

def test_progress_tracking(mock_csv_file):
    """Test progress tracking and callbacks"""
    progress_updates = []
    
    def progress_callback(job):
        progress_updates.append(job.progress)
    
    config = BatchProcessingConfig(
        chunk_size=100,
        progress_callback=progress_callback
    )
    pipeline = BatchPredictionPipeline(config)
    
    content = "dog_name,track,odds\n"
    for i in range(10):
        content += f"Dog{i},MEADOWS,{5.0}\n"
    
    csv_file = mock_csv_file(content)
    
    job_id = pipeline.create_batch_job(
        name="Progress Test",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = pipeline.run_batch_job(job_id)
    
    # Should have received progress updates
    assert len(progress_updates) > 0
    assert job.progress == 100.0
    
    # Clean up
    os.unlink(csv_file)

def test_multiple_file_processing(mock_csv_file, batch_pipeline):
    """Test processing multiple files in batch"""
    content1 = "dog_name,track,odds\nLuna,MEADOWS,5.0\nMax,WENTWORTH,3.0\n"
    content2 = "dog_name,track,odds\nOscar,DAPTO,4.5\nBella,SANDOWN,2.8\n"
    
    csv_file1 = mock_csv_file(content1)
    csv_file2 = mock_csv_file(content2)
    
    job_id = batch_pipeline.create_batch_job(
        name="Multiple Files Test",
        input_files=[csv_file1, csv_file2],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)
    
    # Should process both files
    assert job.total_files == 2
    assert job.completed_files + job.failed_files == 2
    
    # Clean up
    os.unlink(csv_file1)
    os.unlink(csv_file2)

def test_edge_case_dog_name_handling(mock_csv_file, batch_pipeline):
    """Test edge cases in dog name handling (empty names, continuation rows)"""
    content = """dog_name,track,odds,weight
Luna,MEADOWS,5.0,32.5
,,4.8,32.3
Max,WENTWORTH,3.0,33.0
,,2.9,33.2
Oscar,DAPTO,4.5,31.8
"""
    
    csv_file = mock_csv_file(content)
    
    job_id = batch_pipeline.create_batch_job(
        name="Dog Name Edge Cases",
        input_files=[csv_file],
        output_dir="./test_output"
    )
    job = batch_pipeline.run_batch_job(job_id)
    
    # Should handle continuation rows properly
    assert job.status in ['completed', 'completed_with_errors']
    assert job.completed_files == 1
    
    # Clean up
    os.unlink(csv_file)

