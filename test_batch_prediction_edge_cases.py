#!/usr/bin/env python3
"""
Comprehensive Pytest Tests for Batch Prediction Pipeline Edge Cases
==================================================================

Tests cover:
- Empty CSV files and rows
- Unicode handling and international characters
- Mixed track conditions and formats
- Pagination and streaming for large CSV uploads
- Memory exhaustion prevention
- Error handling and recovery
- Progress callback functionality

Author: AI Assistant
Date: December 2024
"""

import csv
import io
import json
import os
import pytest
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Test data fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def batch_pipeline():
    """Create batch prediction pipeline instance"""
    try:
        from batch_prediction_pipeline import BatchPredictionPipeline
        pipeline = BatchPredictionPipeline()
        yield pipeline
        # Cleanup any test jobs
        try:
            for job in pipeline.list_jobs():
                if job.name.startswith("TEST_"):
                    pipeline.cancel_job(job.job_id)
        except:
            pass
    except ImportError:
        pytest.skip("BatchPredictionPipeline not available")

@pytest.fixture
def sample_race_data():
    """Sample race data with various formats"""
    return [
        ["Dog Name", "Box", "Odds", "Weight", "Trainer"],
        ["1. FAST RUNNER", "1", "2.50", "32.5", "J. Smith"],
        ["2. QUICK DASH", "2", "3.80", "31.2", "M. Jones"],
        ["3. SPEED DEMON", "3", "5.20", "33.1", "K. Brown"],
    ]

@pytest.fixture
def unicode_race_data():
    """Race data with unicode characters"""
    return [
        ["Dog Name", "Box", "Odds", "Weight", "Trainer"],
        ["1. BJÖRK'S FURY", "1", "2.50", "32.5", "José García"],
        ["2. MÜLLER'S PRIDE", "2", "3.80", "31.2", "François Müller"],
        ["3. 東京スピード", "3", "5.20", "33.1", "山田太郎"],  # Japanese characters
        ["4. Спринтер", "4", "4.20", "32.8", "Иван Петров"],  # Cyrillic
    ]

@pytest.fixture
def mixed_track_data():
    """Race data with mixed track conditions"""
    return [
        ["Dog Name", "Box", "Odds", "Weight", "Trainer", "Track Condition", "Distance"],
        ["1. FAST RUNNER", "1", "2.50", "32.5", "J. Smith", "Good 4", "500m"],
        ["2. QUICK DASH", "2", "3.80", "31.2", "M. Jones", "Slow 6", "500"],
        ["3. SPEED DEMON", "3", "5.20", "33.1", "K. Brown", "Heavy", "500 metres"],
        ["4. TRACK STAR", "4", "6.10", "30.9", "L. Davis", "GOOD", "0.5km"],
    ]

def create_csv_file(data, file_path, encoding='utf-8'):
    """Helper to create CSV files with specific encoding"""
    with open(file_path, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerows(data)

def create_large_csv_file(file_path, num_rows=10000):
    """Create large CSV file for memory testing"""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Dog Name", "Box", "Odds", "Weight", "Trainer", "Track Condition"])
        
        # Generate many rows
        for i in range(num_rows):
            writer.writerow([
                f"{i+1}. DOG_{i:05d}",
                str((i % 8) + 1),
                f"{2.5 + (i % 20) * 0.1:.2f}",
                f"{30.0 + (i % 50) * 0.1:.1f}",
                f"TRAINER_{i % 100}",
                ["Good", "Slow", "Heavy", "Fast"][i % 4]
            ])

class TestBatchPredictionEdgeCases:
    """Test suite for batch prediction edge cases"""
    
    def test_empty_csv_file(self, batch_pipeline, temp_dir):
        """Test handling of empty CSV files"""
        empty_file = os.path.join(temp_dir, "empty.csv")
        
        # Create completely empty file
        Path(empty_file).touch()
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Empty_File",
            input_files=[empty_file],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        # Run job and check it handles empty file gracefully
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should complete but mark file as failed
        assert job.status in ['completed', 'failed']
        assert job.failed_files == 1
        assert job.completed_files == 0
    
    def test_csv_with_empty_rows(self, batch_pipeline, temp_dir, sample_race_data):
        """Test CSV files with empty rows"""
        csv_file = os.path.join(temp_dir, "empty_rows.csv")
        
        # Add empty rows to sample data
        data_with_empty = sample_race_data.copy()
        data_with_empty.insert(2, [])  # Empty row
        data_with_empty.insert(4, ["", "", "", "", ""])  # Row with empty strings
        data_with_empty.append([])  # Empty row at end
        
        create_csv_file(data_with_empty, csv_file)
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Empty_Rows",
            input_files=[csv_file],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should handle empty rows gracefully
        assert job.status == 'completed'
        
    def test_unicode_handling(self, batch_pipeline, temp_dir, unicode_race_data):
        """Test handling of unicode characters in CSV files"""
        csv_file = os.path.join(temp_dir, "unicode.csv")
        
        # Create file with unicode data
        create_csv_file(unicode_race_data, csv_file, encoding='utf-8')
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Unicode",
            input_files=[csv_file],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should handle unicode characters properly
        assert job.status == 'completed'
        assert job.completed_files == 1
        assert job.failed_files == 0
    
    def test_mixed_track_conditions(self, batch_pipeline, temp_dir, mixed_track_data):
        """Test handling of mixed track condition formats"""
        csv_file = os.path.join(temp_dir, "mixed_tracks.csv")
        
        create_csv_file(mixed_track_data, csv_file)
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Mixed_Tracks",
            input_files=[csv_file],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should normalize and handle different track condition formats
        assert job.status == 'completed'
        assert job.completed_files == 1
    
    def test_large_csv_memory_handling(self, batch_pipeline, temp_dir):
        """Test memory handling with large CSV files"""
        large_csv = os.path.join(temp_dir, "large.csv")
        
        # Create large CSV file (10k rows)
        create_large_csv_file(large_csv, num_rows=10000)
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Large_CSV",
            input_files=[large_csv],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        # Monitor memory usage during processing
        import psutil
        import os as os_module
        
        process = psutil.Process(os_module.getpid())
        memory_before = process.memory_info().rss
        
        batch_pipeline.run_batch_job(job_id)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        job = batch_pipeline.get_job_status(job_id)
        
        # Should complete without excessive memory usage (less than 100MB increase)
        assert job.status == 'completed'
        assert memory_increase < 100 * 1024 * 1024  # 100MB limit
    
    def test_pagination_streaming(self, batch_pipeline, temp_dir):
        """Test pagination/streaming functionality for large datasets"""
        # Create multiple CSV files
        csv_files = []
        for i in range(5):
            csv_file = os.path.join(temp_dir, f"batch_{i}.csv")
            create_large_csv_file(csv_file, num_rows=1000)
            csv_files.append(csv_file)
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Pagination",
            input_files=csv_files,
            output_dir=temp_dir,
            batch_size=2,  # Process 2 files at a time
            max_workers=2
        )
        
        # Monitor progress in background
        progress_updates = []
        
        def monitor_progress():
            while True:
                job = batch_pipeline.get_job_status(job_id)
                if not job:
                    break
                progress_updates.append({
                    'progress': job.progress,
                    'completed': job.completed_files,
                    'total': job.total_files
                })
                if job.status in ['completed', 'failed', 'cancelled']:
                    break
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        batch_pipeline.run_batch_job(job_id)
        monitor_thread.join(timeout=1)
        
        job = batch_pipeline.get_job_status(job_id)
        
        # Should complete with proper pagination
        assert job.status == 'completed'
        assert job.completed_files == 5
        assert len(progress_updates) > 1  # Should have multiple progress updates
    
    def test_progress_callback_functionality(self, batch_pipeline, temp_dir, sample_race_data):
        """Test progress callback functionality"""
        # Create test files
        csv_files = []
        for i in range(3):
            csv_file = os.path.join(temp_dir, f"callback_test_{i}.csv")
            create_csv_file(sample_race_data, csv_file)
            csv_files.append(csv_file)
        
        # Track callback invocations
        callback_data = []
        
        def progress_callback(job_status):
            callback_data.append({
                'progress': job_status.progress,
                'completed': job_status.completed_files,
                'failed': job_status.failed_files,
                'status': job_status.status
            })
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Callbacks",
            input_files=csv_files,
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1,
            progress_callback=progress_callback
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should have multiple callback invocations
        assert job.status == 'completed'
        assert len(callback_data) > 0
        assert callback_data[-1]['progress'] == 100.0  # Final callback should be 100%
    
    def test_malformed_csv_handling(self, batch_pipeline, temp_dir):
        """Test handling of malformed CSV files"""
        malformed_csv = os.path.join(temp_dir, "malformed.csv")
        
        # Create malformed CSV with inconsistent columns
        with open(malformed_csv, 'w', encoding='utf-8') as f:
            f.write("Dog Name,Box,Odds\\n")  # Header with 3 columns
            f.write("1. TEST DOG,1,2.50,EXTRA,DATA\\n")  # Row with 5 columns
            f.write("2. ANOTHER DOG\\n")  # Row with 1 column
            f.write("3. THIRD DOG,3\\n")  # Row with 2 columns
            f.write("Completely invalid line with no commas\\n")
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Malformed",
            input_files=[malformed_csv],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should complete but handle malformed data gracefully
        assert job.status in ['completed', 'failed']
        # May succeed with data cleanup or fail gracefully
        
    def test_concurrent_job_execution(self, batch_pipeline, temp_dir, sample_race_data):
        """Test concurrent execution of multiple batch jobs"""
        # Create multiple sets of CSV files
        job_ids = []
        
        for job_num in range(3):
            csv_files = []
            for file_num in range(2):
                csv_file = os.path.join(temp_dir, f"concurrent_{job_num}_{file_num}.csv")
                create_csv_file(sample_race_data, csv_file)
                csv_files.append(csv_file)
            
            job_id = batch_pipeline.create_batch_job(
                name=f"TEST_Concurrent_{job_num}",
                input_files=csv_files,
                output_dir=temp_dir,
                batch_size=1,
                max_workers=1
            )
            job_ids.append(job_id)
        
        # Start all jobs concurrently
        threads = []
        for job_id in job_ids:
            thread = threading.Thread(
                target=batch_pipeline.run_batch_job,
                args=(job_id,),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all jobs to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Check all jobs completed successfully
        for job_id in job_ids:
            job = batch_pipeline.get_job_status(job_id)
            assert job.status == 'completed'
            assert job.completed_files == 2
    
    def test_job_cancellation(self, batch_pipeline, temp_dir):
        """Test job cancellation functionality"""
        # Create files that would take some time to process
        csv_files = []
        for i in range(5):
            csv_file = os.path.join(temp_dir, f"cancel_test_{i}.csv")
            create_large_csv_file(csv_file, num_rows=5000)
            csv_files.append(csv_file)
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Cancellation",
            input_files=csv_files,
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        # Start job in background
        def run_job():
            batch_pipeline.run_batch_job(job_id)
        
        job_thread = threading.Thread(target=run_job, daemon=True)
        job_thread.start()
        
        # Wait a bit then cancel
        time.sleep(1)
        success = batch_pipeline.cancel_job(job_id)
        
        # Wait for job to stop
        job_thread.join(timeout=5)
        
        job = batch_pipeline.get_job_status(job_id)
        
        # Should successfully cancel
        assert success is True
        assert job.status == 'cancelled'
    
    def test_error_recovery_and_reporting(self, batch_pipeline, temp_dir, sample_race_data):
        """Test error recovery and detailed error reporting"""
        # Create mix of valid and invalid files
        valid_file = os.path.join(temp_dir, "valid.csv")
        create_csv_file(sample_race_data, valid_file)
        
        invalid_file = os.path.join(temp_dir, "invalid.csv")
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write("This is not a CSV file\\n")
            f.write("It contains no proper structure\\n")
        
        missing_file = os.path.join(temp_dir, "missing.csv")  # File doesn't exist
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Error_Recovery",
            input_files=[valid_file, invalid_file, missing_file],
            output_dir=temp_dir,
            batch_size=1,
            max_workers=1
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should complete with some successes and some failures
        assert job.status == 'completed'
        assert job.completed_files == 1  # Only valid file should succeed
        assert job.failed_files == 2  # Invalid and missing files should fail
        assert len(job.error_messages) > 0  # Should have error messages
    
    def test_output_directory_creation(self, batch_pipeline, temp_dir, sample_race_data):
        """Test automatic output directory creation"""
        csv_file = os.path.join(temp_dir, "test.csv")
        create_csv_file(sample_race_data, csv_file)
        
        # Use non-existent output directory
        output_dir = os.path.join(temp_dir, "nested", "output", "directory")
        
        job_id = batch_pipeline.create_batch_job(
            name="TEST_Output_Dir",
            input_files=[csv_file],
            output_dir=output_dir,
            batch_size=1,
            max_workers=1
        )
        
        batch_pipeline.run_batch_job(job_id)
        job = batch_pipeline.get_job_status(job_id)
        
        # Should create output directory and complete successfully
        assert job.status == 'completed'
        assert os.path.exists(output_dir)
    
    def test_encoding_detection_and_handling(self, batch_pipeline, temp_dir, sample_race_data):
        """Test automatic encoding detection for CSV files"""
        # Create files with different encodings
        utf8_file = os.path.join(temp_dir, "utf8.csv")
        create_csv_file(sample_race_data, utf8_file, encoding='utf-8')
        
        # Create file with different encoding (if supported by system)
        try:
            latin1_file = os.path.join(temp_dir, "latin1.csv")
            data_with_accents = [
                ["Dog Name", "Box", "Odds", "Weight", "Trainer"],
                ["1. JOSÉ'S RUNNER", "1", "2.50", "32.5", "José García"],
                ["2. CAFÉ RACER", "2", "3.80", "31.2", "François"],
            ]
            create_csv_file(data_with_accents, latin1_file, encoding='latin-1')
            
            job_id = batch_pipeline.create_batch_job(
                name="TEST_Encoding",
                input_files=[utf8_file, latin1_file],
                output_dir=temp_dir,
                batch_size=1,
                max_workers=1
            )
            
            batch_pipeline.run_batch_job(job_id)
            job = batch_pipeline.get_job_status(job_id)
            
            # Should handle both encodings
            assert job.status == 'completed'
            assert job.completed_files == 2
            
        except UnicodeEncodeError:
            # Skip if system doesn't support latin-1 encoding
            pytest.skip("System doesn't support latin-1 encoding")

class TestBatchPipelineIntegration:
    """Integration tests for batch prediction pipeline"""
    
    def test_cli_integration(self, temp_dir, sample_race_data):
        """Test CLI integration with batch pipeline"""
        # Create test CSV file
        csv_file = os.path.join(temp_dir, "cli_test.csv")
        create_csv_file(sample_race_data, csv_file)
        
        # Test CLI script exists and is executable
        cli_script = "cli_batch_predictor.py"
        assert os.path.exists(cli_script)
        
        # Test help command
        import subprocess
        result = subprocess.run(
            [sys.executable, cli_script, "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "CLI Batch Predictor" in result.stdout
    
    def test_api_endpoint_integration(self, batch_pipeline, temp_dir, sample_race_data):
        """Test API endpoint integration"""
        # This would test the Flask API endpoints
        # For now, we'll test the underlying pipeline methods
        
        csv_file = os.path.join(temp_dir, "api_test.csv")
        create_csv_file(sample_race_data, csv_file)
        
        # Test creating job via API-like interface
        job_data = {
            "name": "TEST_API_Integration",
            "input_files": [csv_file],
            "output_dir": temp_dir,
            "batch_size": 1,
            "max_workers": 1
        }
        
        job_id = batch_pipeline.create_batch_job(**job_data)
        assert job_id is not None
        
        # Test job status retrieval
        job = batch_pipeline.get_job_status(job_id)
        assert job is not None
        assert job.name == "TEST_API_Integration"
        
        # Test job execution
        batch_pipeline.run_batch_job(job_id)
        final_job = batch_pipeline.get_job_status(job_id)
        assert final_job.status == 'completed'

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
