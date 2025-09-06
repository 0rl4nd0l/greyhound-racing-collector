#!/usr/bin/env python3
"""
Example Usage of Structured Logging Configuration
=================================================

This file demonstrates how to use the new structured logging system
that routes messages to appropriate component-specific logs.

Author: AI Assistant
Date: August 4, 2025
"""

from config.logging_config import (
    get_component_logger,
    log_config,
    log_gpt_assistant,
    log_prediction,
    log_qa,
    log_test,
)


def example_prediction_logging():
    """Example of prediction-related logging"""
    print("📊 Example: Prediction logging")

    # Model training logs
    log_prediction(
        "Starting model training pipeline",
        level="INFO",
        action="train_model",
        outcome="started",
        details={
            "model_type": "ExtraTreesClassifier",
            "version": "v4",
            "dataset_size": 10000,
        },
    )

    log_prediction(
        "Model training completed",
        level="INFO",
        action="train_model",
        outcome="success",
        cache_status="stored",
        details={"accuracy": 0.85, "training_time": "2.5 minutes"},
    )

    # Inference logs
    log_prediction(
        "Running batch prediction",
        level="INFO",
        action="batch_predict",
        outcome="in_progress",
        details={"batch_size": 100, "input_source": "race_data.csv"},
    )


def example_test_logging():
    """Example of test-related logging"""
    print("🧪 Example: Test logging")

    log_test(
        "Running comprehensive test suite",
        level="INFO",
        action="run_tests",
        outcome="started",
        details={
            "test_types": ["unit", "integration", "regression"],
            "total_tests": 250,
        },
    )

    log_test(
        "Test suite completed",
        level="INFO",
        action="run_tests",
        outcome="success",
        validation_errors=[],
        details={"passed": 248, "failed": 2, "skipped": 0},
    )

    # Error case
    log_test(
        "Test failure detected",
        level="WARNING",
        action="test_failure",
        outcome="failed",
        validation_errors=["test_prediction_accuracy: AssertionError"],
        details={
            "test_name": "test_prediction_accuracy",
            "expected": 0.8,
            "actual": 0.75,
        },
    )


def example_qa_logging():
    """Example of QA-related logging"""
    print("🔍 Example: QA logging")

    log_qa(
        "Starting data integrity check",
        level="INFO",
        action="integrity_check",
        outcome="started",
        details={
            "tables_to_check": ["races", "dogs", "results"],
            "check_type": "full_scan",
        },
    )

    log_qa(
        "Data integrity check completed",
        level="INFO",
        action="integrity_check",
        outcome="success",
        validation_errors=[],
        details={"records_validated": 50000, "errors_found": 0},
    )

    # QA issue detected
    log_qa(
        "Data quality issue detected",
        level="WARNING",
        action="quality_check",
        outcome="issues_found",
        validation_errors=["Missing dog names in 5 records"],
        details={"affected_records": 5, "severity": "medium"},
    )


def example_gpt_assistant_logging():
    """Example of GPT assistant-related logging"""
    print("🤖 Example: GPT Assistant logging")

    log_gpt_assistant(
        "Processing GPT enhancement request",
        level="INFO",
        action="gpt_enhance",
        outcome="processing",
        details={"request_type": "prediction_analysis", "model": "gpt-4"},
    )

    log_gpt_assistant(
        "GPT analysis completed",
        level="INFO",
        action="gpt_enhance",
        outcome="success",
        cache_status="cached",
        details={"tokens_used": 1500, "response_length": 800},
    )


def example_config_logging():
    """Example of configuration-related logging"""
    print("⚙️ Example: Configuration logging")

    log_config(
        "Loading system configuration",
        level="INFO",
        action="config_load",
        outcome="started",
        details={"config_files": ["defaults.yaml", "logging_config.py"]},
    )

    log_config(
        "Configuration validation passed",
        level="INFO",
        action="config_validate",
        outcome="success",
        validation_errors=[],
        details={"validated_sections": ["logging", "database", "ml_models"]},
    )


def example_component_detection():
    """Example of automatic component detection based on context"""
    print("🎯 Example: Automatic component detection")

    # Get the component logger directly
    logger = get_component_logger()

    # These will be automatically routed based on keywords in action/details
    logger.info(
        "Model calibration started",
        action="calibrate_model",  # 'calibrat' -> prediction
        details={"calibration_method": "isotonic"},
    )

    logger.info(
        "Running unit tests for data validation",
        action="validate_data",  # 'validate' -> test
        details={"test_framework": "pytest"},
    )

    logger.info(
        "Audit trail generated",
        action="audit_trail",  # 'audit' -> qa
        details={"audit_type": "security_review"},
    )

    logger.info(
        "AI enhancement processing",
        action="ai_process",  # 'ai' -> gpt_assistant
        details={"enhancement_type": "prediction_improvement"},
    )

    logger.info(
        "System initialization complete",
        action="init_system",  # 'init' -> config
        details={"modules_loaded": ["logging", "database", "ml"]},
    )


def example_debug_logging():
    """Example of debug-level logging (only appears when debug mode is enabled)"""
    print("🐛 Example: Debug logging")

    # Debug logs - only appear when DEBUG=1 environment variable is set or --debug flag is used
    log_prediction(
        "Debug: Feature extraction details",
        level="DEBUG",
        action="extract_features",
        outcome="debug_info",
        details={"features_extracted": 33, "processing_time": 0.05},
    )

    log_test(
        "Debug: Test setup information",
        level="DEBUG",
        action="test_setup",
        outcome="debug_info",
        details={"fixtures_loaded": ["database", "sample_data"], "setup_time": 0.1},
    )


if __name__ == "__main__":
    print("🚀 Structured Logging Examples")
    print("=" * 50)

    example_prediction_logging()
    print()

    example_test_logging()
    print()

    example_qa_logging()
    print()

    example_gpt_assistant_logging()
    print()

    example_config_logging()
    print()

    example_component_detection()
    print()

    example_debug_logging()
    print()

    print("✅ All examples completed!")
    print("📁 Check the following directories for log files:")
    print("   - logs/prediction/")
    print("   - logs/test/")
    print("   - logs/qa/")
    print("   - gpt_assistant/")
    print("   - config/")
    print("   - logs/main_workflow.jsonl (contains all logs)")
