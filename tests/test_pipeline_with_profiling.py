import pytest
from prediction_pipeline_v3 import PredictionPipelineV3
from unittest.mock import MagicMock
from profiling_config import set_profiling_enabled, is_profiling
from utils.profiling_utils import ProfilingRecorder


# Mocks and constants
def mock_pipeline_predict(mocker):
    mocked_predict = MagicMock()
    mocked_predict.return_value = {'success': True, 'predictions': []}
    mocker.patch('prediction_pipeline_v3.MLSystemV3').predict = mocked_predict


def test_pipeline_with_profiling_on(mocker):
    """Test pipeline with profiling enabled"""
    set_profiling_enabled(True)
    assert is_profiling() is True

    mock_pipeline_predict(mocker)

    pipeline = PredictionPipelineV3()
    result = pipeline.predict_race_file('path_to_non_existent_file.csv')  # Replace with actual paths
    
    ProfilingRecorder.end_session()

    assert result['success'] is True  # Ensuring prediction returns success
    
    # Check profiling data (Mock this part based on actual structure)
    assert ProfilingRecorder.recordings  # Ensuring recordings are present


def test_pipeline_with_profiling_off(mocker):
    """Test pipeline with profiling disabled"""
    set_profiling_enabled(False)
    assert is_profiling() is False

    mock_pipeline_predict(mocker)

    pipeline = PredictionPipelineV3()
    result = pipeline.predict_race_file('path_to_non_existent_file.csv')  # Replace with actual paths

    assert result['success'] is True
    assert not ProfilingRecorder.recordings  # No recordings should be made

    set_profiling_enabled(False)
