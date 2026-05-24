"""
Interface definitions for GPTAssistant system components.
This file defines the contracts for all GPTAssistant components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from models import AdvisoryMessage, GuideIssue, PredictionRecord


class LogIngestor(ABC):
    """
    Interface for log ingestion and parsing components.

    Responsible for reading prediction logs from various sources and
    converting them into structured PredictionRecord objects.
    """

    @abstractmethod
    def ingest_logs(self, path: str) -> List[PredictionRecord]:
        """
        Ingest logs from the specified path and return parsed records.

        Args:
            path: File path or directory path to log files

        Returns:
            List of parsed PredictionRecord objects

        Raises:
            FileNotFoundError: If the specified path doesn't exist
            ValueError: If log format is invalid or unsupported
        """
        pass

    @abstractmethod
    def parse_log_entry(self, entry: str) -> Optional[PredictionRecord]:
        """
        Parse a single log entry into a PredictionRecord.

        Args:
            entry: Raw log entry string

        Returns:
            PredictionRecord if parsing successful, None if entry is invalid
        """
        pass

    @abstractmethod
    def validate_format(self, entry: str) -> bool:
        """
        Validate if a log entry follows the expected format.

        Args:
            entry: Raw log entry string

        Returns:
            True if format is valid, False otherwise
        """
        pass


class FormGuideValidator(ABC):
    """
    Interface for form guide data validation components.

    Responsible for validating the completeness and consistency of
    form guide data in prediction records.
    """

    @abstractmethod
    def validate_form_data(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """
        Validate form data across multiple prediction records.

        Args:
            records: List of prediction records to validate

        Returns:
            List of GuideIssue objects representing validation problems
        """
        pass

    @abstractmethod
    def check_data_completeness(self, record: PredictionRecord) -> List[GuideIssue]:
        """
        Check if a single prediction record has complete form data.

        Args:
            record: PredictionRecord to check for completeness

        Returns:
            List of GuideIssue objects for missing or incomplete data
        """
        pass

    @abstractmethod
    def validate_consistency(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """
        Validate consistency of form data across multiple records.

        Args:
            records: List of prediction records to check for consistency

        Returns:
            List of GuideIssue objects for consistency violations
        """
        pass


class PredictionQAAnalyzer(ABC):
    """
    Interface for prediction quality analysis components.

    Responsible for analyzing prediction quality, detecting anomalies,
    and identifying potential performance issues.
    """

    @abstractmethod
    def analyze_predictions(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """
        Perform comprehensive analysis of prediction records.

        Args:
            records: List of prediction records to analyze

        Returns:
            List of GuideIssue objects representing analysis findings
        """
        pass

    @abstractmethod
    def check_accuracy_patterns(
        self, records: List[PredictionRecord]
    ) -> List[GuideIssue]:
        """
        Analyze accuracy patterns in prediction records.

        Args:
            records: List of prediction records to analyze

        Returns:
            List of GuideIssue objects for accuracy-related issues
        """
        pass

    @abstractmethod
    def detect_anomalies(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """
        Detect statistical anomalies in prediction data.

        Args:
            records: List of prediction records to analyze

        Returns:
            List of GuideIssue objects for detected anomalies
        """
        pass


class AdvisoryDispatcher(ABC):
    """
    Interface for advisory generation and dispatch components.

    Responsible for converting issues into actionable advisory messages
    and coordinating their delivery to output reporters.
    """

    @abstractmethod
    def generate_advisories(self, issues: List[GuideIssue]) -> List[AdvisoryMessage]:
        """
        Generate advisory messages from a list of issues.

        Args:
            issues: List of GuideIssue objects to convert

        Returns:
            List of AdvisoryMessage objects
        """
        pass

    @abstractmethod
    def prioritize_issues(self, issues: List[GuideIssue]) -> List[GuideIssue]:
        """
        Prioritize issues based on severity and impact.

        Args:
            issues: List of GuideIssue objects to prioritize

        Returns:
            List of GuideIssue objects sorted by priority
        """
        pass

    @abstractmethod
    def create_advisory(self, issue: GuideIssue) -> AdvisoryMessage:
        """
        Create a single advisory message from an issue.

        Args:
            issue: GuideIssue to convert to advisory

        Returns:
            AdvisoryMessage object
        """
        pass

    @abstractmethod
    def dispatch_to_reporters(self, advisories: List[AdvisoryMessage]) -> None:
        """
        Dispatch advisory messages to configured output reporters.

        Args:
            advisories: List of AdvisoryMessage objects to dispatch
        """
        pass


class OutputReporter(ABC):
    """
    Interface for output reporting components.

    Responsible for formatting and delivering advisory messages
    to various output destinations.
    """

    @abstractmethod
    def format_advisory(self, advisory: AdvisoryMessage) -> str:
        """
        Format an advisory message for output.

        Args:
            advisory: AdvisoryMessage to format

        Returns:
            Formatted string representation
        """
        pass

    @abstractmethod
    def output_advisories(self, advisories: List[AdvisoryMessage]) -> bool:
        """
        Output a list of advisory messages.

        Args:
            advisories: List of AdvisoryMessage objects to output

        Returns:
            True if output was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_output_destination(self) -> str:
        """
        Get the output destination identifier.

        Returns:
            String identifying the output destination
        """
        pass


class GPTAssistant:
    """
    Main GPTAssistant class that orchestrates the analysis pipeline.

    This class coordinates the flow from log ingestion through analysis
    to advisory generation and output reporting.
    """

    def __init__(
        self,
        log_ingestor: LogIngestor,
        form_validator: FormGuideValidator,
        qa_analyzer: PredictionQAAnalyzer,
        advisory_dispatcher: AdvisoryDispatcher,
        reporters: List[OutputReporter],
    ):
        """
        Initialize GPTAssistant with required components.

        Args:
            log_ingestor: Component for log ingestion and parsing
            form_validator: Component for form data validation
            qa_analyzer: Component for prediction quality analysis
            advisory_dispatcher: Component for advisory generation
            reporters: List of output reporter components
        """
        self.log_ingestor = log_ingestor
        self.form_validator = form_validator
        self.qa_analyzer = qa_analyzer
        self.advisory_dispatcher = advisory_dispatcher
        self.reporters = reporters

    def run_analysis(self, log_path: str) -> bool:
        """
        Run the complete analysis pipeline.

        Args:
            log_path: Path to log files to analyze

        Returns:
            True if analysis completed successfully, False otherwise
        """
        try:
            # Step 1: Ingest logs
            records = self.log_ingestor.ingest_logs(log_path)
            if not records:
                print(f"No records found in {log_path}")
                return False

            # Step 2: Validate form data and analyze predictions in parallel
            form_issues = self.form_validator.validate_form_data(records)
            qa_issues = self.qa_analyzer.analyze_predictions(records)

            # Step 3: Combine and prioritize issues
            all_issues = form_issues + qa_issues
            if not all_issues:
                print("No issues detected")
                return True

            # Step 4: Generate and dispatch advisories
            advisories = self.advisory_dispatcher.generate_advisories(all_issues)
            self.advisory_dispatcher.dispatch_to_reporters(advisories)

            # Step 5: Output to all configured reporters
            for reporter in self.reporters:
                reporter.output_advisories(advisories)

            return True

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return False

    def process_logs(self, log_path: str) -> List[AdvisoryMessage]:
        """
        Process logs and return advisory messages without outputting them.

        Args:
            log_path: Path to log files to analyze

        Returns:
            List of AdvisoryMessage objects generated from analysis
        """
        records = self.log_ingestor.ingest_logs(log_path)
        form_issues = self.form_validator.validate_form_data(records)
        qa_issues = self.qa_analyzer.analyze_predictions(records)
        all_issues = form_issues + qa_issues

        return self.advisory_dispatcher.generate_advisories(all_issues)

    def generate_report(self, advisories: List[AdvisoryMessage]) -> str:
        """
        Generate a summary report from advisory messages.

        Args:
            advisories: List of AdvisoryMessage objects

        Returns:
            Formatted summary report string
        """
        if not advisories:
            return "No issues detected in analysis."

        report_lines = [
            "=== GPTAssistant Analysis Report ===",
            f"Total Issues Found: {len(advisories)}",
            "",
        ]

        # Group by severity
        severity_counts = {}
        for advisory in advisories:
            severity = advisory.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report_lines.append("Issues by Severity:")
        for severity, count in sorted(severity_counts.items()):
            report_lines.append(f"  {severity.upper()}: {count}")

        report_lines.extend(["", "=== Advisory Details ===", ""])

        for i, advisory in enumerate(advisories, 1):
            report_lines.extend(
                [
                    f"{i}. {advisory.title} [{advisory.severity.value.upper()}]",
                    f"   Summary: {advisory.summary}",
                    f"   Action Items: {', '.join(advisory.action_items)}",
                    "",
                ]
            )

        return "\n".join(report_lines)
