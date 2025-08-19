#!/usr/bin/env python3
"""
Validation & Quarantine Workflow - Step 4
==========================================

This module implements the validation and quarantine workflow as specified:
• Inspect ValidationReport after parsing
• Move files to ./quarantine with YYYY-MM-DD subdirectories if critical errors exceed threshold
• Continue to DB insert/prediction if validation passes
• Emit structured JSON log entries with validation statistics

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing parsing components
from enhanced_form_guide_parser import EnhancedFormGuideParser, ValidationIssue, ValidationSeverity, ParsingResult


class ValidationDecision(Enum):
    """Validation decision outcomes"""
    CONTINUE_TO_DB = "continue_to_db"
    QUARANTINE_CRITICAL = "quarantine_critical"
    QUARANTINE_THRESHOLD = "quarantine_threshold"


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    file_path: str
    timestamp: str
    processing_decision: ValidationDecision
    total_issues: int
    critical_errors: int
    warnings: int
    info_messages: int
    data_records_count: int
    quarantine_reason: Optional[str] = None
    quarantine_path: Optional[str] = None
    validation_statistics: Dict[str, Any] = None
    

@dataclass
class QuarantineConfig:
    """Configuration for quarantine system"""
    base_quarantine_dir: str = "./quarantine"
    critical_error_threshold: int = 5  # Max critical errors before quarantine
    critical_error_percentage: float = 0.3  # Max 30% critical errors
    use_date_subdirectories: bool = True
    

class ValidationQuarantineWorkflow:
    """
    Main workflow controller for validation and quarantine processing.
    
    Integrates with existing parsing system and implements the spec requirements:
    1. Parse file using enhanced parser
    2. Inspect ValidationReport for critical errors  
    3. If errors exceed threshold -> quarantine with YYYY-MM-DD structure
    4. Else continue to DB insert/prediction
    5. Emit structured JSON logs
    """
    
    def __init__(self, config: QuarantineConfig = None):
        """
        Initialize the validation workflow.
        
        Args:
            config: Configuration for quarantine behavior
        """
        self.config = config or QuarantineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced parser
        self.parser = EnhancedFormGuideParser(quarantine_dir=self.config.base_quarantine_dir)
        
        # Ensure quarantine directory exists
        self.quarantine_base = Path(self.config.base_quarantine_dir)
        self.quarantine_base.mkdir(exist_ok=True)
        
        # Setup structured logging
        self._setup_structured_logging()
        
    def _setup_structured_logging(self):
        """Setup structured JSON logging for validation events"""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create validation log handler
        validation_log_path = Path("validation_workflow.log")
        file_handler = logging.FileHandler(validation_log_path)
        file_handler.setFormatter(log_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def _count_critical_errors(self, issues: List[ValidationIssue]) -> Tuple[int, int, int]:
        """
        Count issues by severity level.
        
        Args:
            issues: List of validation issues
            
        Returns:
            Tuple of (critical_errors, warnings, info_messages)
        """
        critical_errors = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        info_messages = sum(1 for issue in issues if issue.severity == ValidationSeverity.INFO)
        
        return critical_errors, warnings, info_messages
        
    def _determine_validation_decision(self, parsing_result: ParsingResult, file_path: Path) -> Tuple[ValidationDecision, Optional[str]]:
        """
        Determine whether to continue to DB or quarantine based on validation results.
        
        Args:
            parsing_result: Result from enhanced parser
            file_path: Path to the file being processed
            
        Returns:
            Tuple of (decision, quarantine_reason)
        """
        critical_errors, warnings, info_messages = self._count_critical_errors(parsing_result.issues)
        total_records = len(parsing_result.data)
        
        # Check if already quarantined by parser
        if parsing_result.quarantined:
            return ValidationDecision.QUARANTINE_CRITICAL, "Parser determined file should be quarantined"
            
        # Check critical error threshold (absolute count)
        if critical_errors > self.config.critical_error_threshold:
            reason = f"Critical errors ({critical_errors}) exceed threshold ({self.config.critical_error_threshold})"
            return ValidationDecision.QUARANTINE_THRESHOLD, reason
            
        # Check critical error percentage (if we have records)
        if total_records > 0:
            error_percentage = critical_errors / total_records
            if error_percentage > self.config.critical_error_percentage:
                reason = f"Critical error rate ({error_percentage:.1%}) exceeds threshold ({self.config.critical_error_percentage:.1%})"
                return ValidationDecision.QUARANTINE_THRESHOLD, reason
                
        # If no critical issues, continue to DB
        return ValidationDecision.CONTINUE_TO_DB, None
        
    def _create_date_quarantine_directory(self) -> Path:
        """
        Create quarantine subdirectory with YYYY-MM-DD format for easy triage.
        
        Returns:
            Path to the date-specific quarantine directory
        """
        if self.config.use_date_subdirectories:
            date_str = datetime.now().strftime("%Y-%m-%d")
            quarantine_date_dir = self.quarantine_base / date_str
            quarantine_date_dir.mkdir(exist_ok=True)
            return quarantine_date_dir
        else:
            return self.quarantine_base
            
    def _quarantine_file(self, file_path: Path, parsing_result: ParsingResult, reason: str) -> str:
        """
        Move file to quarantine directory with proper structure and metadata.
        
        Args:
            file_path: Original file path
            parsing_result: Parsing result with validation issues
            reason: Reason for quarantine
            
        Returns:
            Path to quarantined file
        """
        # Create date-specific quarantine directory
        quarantine_dir = self._create_date_quarantine_directory()
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%H%M%S")
        quarantine_filename = f"{timestamp}_{file_path.name}"
        quarantine_file_path = quarantine_dir / quarantine_filename
        
        # Copy file to quarantine (preserve original)
        shutil.copy2(file_path, quarantine_file_path)
        
        # Create detailed quarantine report
        quarantine_report = {
            "original_file": str(file_path),
            "quarantine_timestamp": datetime.now().isoformat(),
            "quarantine_reason": reason,
            "validation_issues": [
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "suggested_fix": issue.suggested_fix
                }
                for issue in parsing_result.issues
            ],
            "issue_summary": {
                "total_issues": len(parsing_result.issues),
                "critical_errors": sum(1 for issue in parsing_result.issues if issue.severity == ValidationSeverity.ERROR),
                "warnings": sum(1 for issue in parsing_result.issues if issue.severity == ValidationSeverity.WARNING),
                "info_messages": sum(1 for issue in parsing_result.issues if issue.severity == ValidationSeverity.INFO)
            },
            "parsing_statistics": parsing_result.statistics,
            "data_records_found": len(parsing_result.data)
        }
        
        # Save quarantine report
        report_path = quarantine_file_path.with_suffix('.quarantine_report.json')
        with open(report_path, 'w') as f:
            json.dump(quarantine_report, f, indent=2)
            
        self.logger.warning(f"File quarantined: {file_path} -> {quarantine_file_path} (Reason: {reason})")
        
        return str(quarantine_file_path)
        
    def _emit_validation_log(self, validation_report: ValidationReport):
        """
        Emit structured JSON log entry with validation statistics.
        
        Args:
            validation_report: Validation report to log
        """
        log_entry = {
            "event_type": "validation_workflow",
            "timestamp": validation_report.timestamp,
            "file_path": validation_report.file_path,
            "processing_decision": validation_report.processing_decision.value,
            "validation_statistics": {
                "total_issues": validation_report.total_issues,
                "critical_errors": validation_report.critical_errors,
                "warnings": validation_report.warnings,
                "info_messages": validation_report.info_messages,
                "data_records_count": validation_report.data_records_count
            },
            "quarantine_info": {
                "quarantined": validation_report.processing_decision != ValidationDecision.CONTINUE_TO_DB,
                "quarantine_reason": validation_report.quarantine_reason,
                "quarantine_path": validation_report.quarantine_path
            } if validation_report.quarantine_reason else None,
            "additional_statistics": validation_report.validation_statistics
        }
        
        # Log as structured JSON
        self.logger.info(f"VALIDATION_WORKFLOW: {json.dumps(log_entry)}")
        
    def process_file(self, file_path: Union[str, Path]) -> ValidationReport:
        """
        Main entry point for validation and quarantine workflow.
        
        This method implements the complete Step 4 workflow:
        1. Parse file using enhanced parser
        2. Inspect validation results for critical errors
        3. If errors exceed threshold -> quarantine with date structure
        4. Else continue to DB insert/prediction
        5. Emit structured JSON log
        
        Args:
            file_path: Path to file to process
            
        Returns:
            ValidationReport with processing decision and details
        """
        file_path = Path(file_path)
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Starting validation workflow for: {file_path}")
        
        try:
            # Step 1: Parse file using enhanced parser
            parsing_result = self.parser.parse_form_guide(file_path)
            
            # Step 2: Count issues and determine decision
            critical_errors, warnings, info_messages = self._count_critical_errors(parsing_result.issues)
            decision, quarantine_reason = self._determine_validation_decision(parsing_result, file_path)
            
            # Step 3: Handle quarantine if needed
            quarantine_path = None
            if decision != ValidationDecision.CONTINUE_TO_DB:
                quarantine_path = self._quarantine_file(file_path, parsing_result, quarantine_reason)
                
            # Create validation report
            validation_report = ValidationReport(
                file_path=str(file_path),
                timestamp=timestamp,
                processing_decision=decision,
                total_issues=len(parsing_result.issues),
                critical_errors=critical_errors,
                warnings=warnings,
                info_messages=info_messages,
                data_records_count=len(parsing_result.data),
                quarantine_reason=quarantine_reason,
                quarantine_path=quarantine_path,
                validation_statistics=parsing_result.statistics
            )
            
            # Step 4: Emit structured log
            self._emit_validation_log(validation_report)
            
            # Step 5: Log summary
            if decision == ValidationDecision.CONTINUE_TO_DB:
                self.logger.info(f"✅ File validation passed - continuing to DB: {file_path}")
                self.logger.info(f"   📊 Records: {len(parsing_result.data)}, Issues: {len(parsing_result.issues)} ({critical_errors} critical)")
            else:
                self.logger.warning(f"🚨 File quarantined: {file_path}")
                self.logger.warning(f"   📊 Reason: {quarantine_reason}")
                self.logger.warning(f"   📁 Quarantine location: {quarantine_path}")
                
            return validation_report
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during validation workflow: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            # Create error validation report
            validation_report = ValidationReport(
                file_path=str(file_path),
                timestamp=timestamp,
                processing_decision=ValidationDecision.QUARANTINE_CRITICAL,
                total_issues=1,
                critical_errors=1,
                warnings=0,
                info_messages=0,
                data_records_count=0,
                quarantine_reason=error_msg,
                quarantine_path=None,
                validation_statistics={}
            )
            
            self._emit_validation_log(validation_report)
            return validation_report
            
    def batch_process_files(self, file_paths: List[Union[str, Path]]) -> List[ValidationReport]:
        """
        Process multiple files through the validation workflow.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of validation reports
        """
        results = []
        
        self.logger.info(f"🚀 Starting batch validation workflow for {len(file_paths)} files")
        
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                
        # Generate batch summary
        total_files = len(results)
        quarantined_files = sum(1 for r in results if r.processing_decision != ValidationDecision.CONTINUE_TO_DB)
        success_files = total_files - quarantined_files
        
        batch_summary = {
            "event_type": "batch_validation_summary",
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "successful_files": success_files,
            "quarantined_files": quarantined_files,
            "success_rate": (success_files / total_files * 100) if total_files > 0 else 0,
            "quarantine_directories_used": list(set(
                Path(r.quarantine_path).parent.name for r in results 
                if r.quarantine_path
            ))
        }
        
        self.logger.info(f"BATCH_VALIDATION_SUMMARY: {json.dumps(batch_summary)}")
        self.logger.info(f"✅ Batch processing complete: {success_files}/{total_files} files passed validation")
        
        return results
        
    def get_quarantine_summary(self) -> Dict[str, Any]:
        """
        Get summary of quarantined files organized by date.
        
        Returns:
            Dictionary with quarantine statistics
        """
        quarantine_summary = {
            "quarantine_base_directory": str(self.quarantine_base),
            "date_directories": {},
            "total_quarantined_files": 0
        }
        
        if not self.quarantine_base.exists():
            return quarantine_summary
            
        # Scan quarantine directories
        for date_dir in self.quarantine_base.iterdir():
            if date_dir.is_dir() and date_dir.name.match(r'\d{4}-\d{2}-\d{2}'):
                # Count files in this date directory
                csv_files = list(date_dir.glob("*.csv"))
                report_files = list(date_dir.glob("*.quarantine_report.json"))
                
                quarantine_summary["date_directories"][date_dir.name] = {
                    "csv_files": len(csv_files),
                    "report_files": len(report_files),
                    "directory_path": str(date_dir)
                }
                
                quarantine_summary["total_quarantined_files"] += len(csv_files)
                
        return quarantine_summary


def main():
    """CLI entry point for validation workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation & Quarantine Workflow")
    parser.add_argument("files", nargs="+", help="CSV files to process")
    parser.add_argument("--quarantine-dir", default="./quarantine", help="Base quarantine directory")
    parser.add_argument("--critical-threshold", type=int, default=5, help="Critical error threshold")
    parser.add_argument("--critical-percentage", type=float, default=0.3, help="Critical error percentage threshold")
    parser.add_argument("--no-date-dirs", action="store_true", help="Don't use date subdirectories")
    parser.add_argument("--summary", action="store_true", help="Show quarantine summary")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = QuarantineConfig(
        base_quarantine_dir=args.quarantine_dir,
        critical_error_threshold=args.critical_threshold,
        critical_error_percentage=args.critical_percentage,
        use_date_subdirectories=not args.no_date_dirs
    )
    
    # Initialize workflow
    workflow = ValidationQuarantineWorkflow(config)
    
    if args.summary:
        # Show quarantine summary
        summary = workflow.get_quarantine_summary()
        print("\n📊 QUARANTINE SUMMARY")
        print("=" * 50)
        print(f"Base directory: {summary['quarantine_base_directory']}")
        print(f"Total quarantined files: {summary['total_quarantined_files']}")
        print("\nBy date:")
        for date, info in summary['date_directories'].items():
            print(f"  {date}: {info['csv_files']} files ({info['report_files']} reports)")
        return
        
    # Process files
    file_paths = [Path(f) for f in args.files]
    
    # Validate files exist
    for file_path in file_paths:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
            
    # Process files
    results = workflow.batch_process_files(file_paths)
    
    # Print results summary
    print(f"\n📋 PROCESSING SUMMARY")
    print("=" * 50)
    for result in results:
        status = "✅ PASS" if result.processing_decision == ValidationDecision.CONTINUE_TO_DB else "🚨 QUARANTINE"
        print(f"{status} - {Path(result.file_path).name}")
        if result.quarantine_reason:
            print(f"    Reason: {result.quarantine_reason}")
            
    success_count = sum(1 for r in results if r.processing_decision == ValidationDecision.CONTINUE_TO_DB)
    print(f"\nOverall: {success_count}/{len(results)} files passed validation")


if __name__ == "__main__":
    main()
